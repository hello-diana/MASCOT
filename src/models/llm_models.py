"""LLM model wrappers for local and API-based inference."""

import json
import os
import re
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Union, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import AzureOpenAI, OpenAI

from src.const import DEFAULT_SPEAKER_MODEL
from src.utils.text import extract_final_response


logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Structured output schema for agent responses."""
    response: str

    @classmethod
    def json_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "response": {"type": "string"}
            },
            "required": ["response"]
        }


def build_chat_messages(prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Canonical chat-message list used by BOTH vLLM backends.

    Local (VLLMModel) and server (VLLMServerModel) generation must format the
    same logical input identically: this is the single place messages are
    assembled. The local backend renders these messages with the tokenizer's
    chat template; the server backend sends them to /chat/completions, where
    the server applies the same template.
    """
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def parse_json_response(text: str) -> Optional[Dict]:
    """Try to parse a JSON object from model response text."""
    text = extract_final_response(text, strip_without_tag=False)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    return None


class VLLMModel:
    """Local vLLM inference wrapper."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.7,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        enable_thinking: bool = True,
        enable_lora: bool = False,
        lora_path: Optional[str] = None,
        max_lora_rank: int = 64,
    ):
        from vllm import LLM, SamplingParams  # noqa: F811

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.lora_path = lora_path

        llm_kwargs = dict(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        if enable_lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max_lora_rank

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _build_sampling_params(self, temperature, top_p, max_tokens,
                               enable_thinking=None, response_format=None):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        if enable_thinking is None:
            enable_thinking = self.enable_thinking
        # Qwen3 thinking mode requires temperature >= 0.6
        if enable_thinking and temperature < 0.6:
            temperature = 0.6

        kwargs = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        if response_format is not None:
            schema = response_format
            if hasattr(response_format, 'json_schema'):
                schema = response_format.json_schema()
            kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)

        return SamplingParams(**kwargs)

    def _render_chat_prompt(self, prompt: str, system_prompt: Optional[str],
                            enable_thinking: bool) -> str:
        """Render the canonical chat messages with the tokenizer's chat template.

        This mirrors what the vLLM server does for /chat/completions, so local
        and server backends see identically formatted inputs.
        """
        messages = build_chat_messages(prompt, system_prompt)
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        except TypeError:
            # Chat template without an enable_thinking kwarg (non-Qwen3 models)
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def generate_batch(self, prompts: List[str], temperature: float = None, top_p: float = None,
                       max_tokens: int = None, enable_thinking: bool = None,
                       response_format=None, lora_request=None,
                       desc: str = "Generating", system_prompt: Optional[str] = None) -> List[str]:
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking

        params = self._build_sampling_params(temperature, top_p, max_tokens,
                                             enable_thinking, response_format)

        if lora_request is None and self.lora_path is not None:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("sft_adapter", 1, self.lora_path)

        # Same logical messages + same chat template as the server backend.
        rendered = [self._render_chat_prompt(p, system_prompt, thinking) for p in prompts]

        logger.info(f"[vLLM Local] {desc} ({len(prompts)} prompts) ...")
        outputs = self.llm.generate(rendered, params, use_tqdm=True, lora_request=lora_request)
        logger.info(f"[vLLM Local] {desc} complete: {len(outputs)} responses")
        return [output.outputs[0].text for output in outputs]



class VLLMServerModel:
    """Connects to an external vLLM server via OpenAI-compatible API.

    This avoids in-process model loading (30-300s) by connecting to a
    persistent vLLM server started separately (see scripts/serve_vllm.py,
    e.g. `python scripts/serve_vllm.py --model Qwen/Qwen3-8B --gpus 2 --port 8008`).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        max_concurrent_requests: int = 32,
        enable_thinking: bool = True,
        role: Optional[str] = None,
        trace: Optional[list] = None,
    ):
        if not base_url:
            raise ValueError(
                "VLLMServerModel requires an explicit base_url (no localhost "
                "default is baked in); pass the server URL via the CLI flags.")
        self.model_name = model_name
        self.base_url = base_url
        self.max_concurrent_requests = max_concurrent_requests
        self.enable_thinking = enable_thinking
        # Role-call tracing: when `trace` (a shared list) is provided, every
        # request appends {role, requested_model_id, desc, is_lora_alias,
        # completion_tokens, success} — runtime evidence of which model id
        # each role actually called.
        self.role = role
        self.trace = trace
        self.is_lora_alias = False

        # vLLM server doesn't need a real API key
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")

        # Fatal, actionable preflight: verify the server is reachable AND that
        # it actually serves the requested model id (catches typo'd LoRA aliases).
        try:
            served = self.client.models.list()
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach vLLM server at {base_url} "
                f"(model {model_name!r}): {e}") from e
        served_ids = [m.id for m in served.data]
        if model_name not in served_ids:
            raise ConnectionError(
                f"vLLM server at {base_url} does not serve model {model_name!r}; "
                f"served ids: {served_ids}")
        self.lora_parent = None
        # id -> {parent, root}: for LoRA modules vLLM reports parent=<base id>
        # and root=<adapter path> — the base-vs-adapter identity evidence.
        self.served_models = {
            m.id: {"parent": getattr(m, "parent", None),
                   "root": getattr(m, "root", None)}
            for m in served.data
        }
        for m in served.data:
            if m.id == model_name:
                self.lora_parent = getattr(m, "parent", None)
                self.is_lora_alias = self.lora_parent is not None
        logger.info(f"Connected to vLLM server at {base_url} "
                    f"(model: {model_name}, lora_alias: {self.is_lora_alias})")

    def _record_trace(self, desc, completion_tokens, success):
        if self.trace is None:
            return
        self.trace.append({
            "role": self.role,
            "requested_model_id": self.model_name,
            "desc": desc,
            "is_lora_alias": self.is_lora_alias,
            "completion_tokens": completion_tokens,
            "success": success,
        })
        
    def generate_batch(
        self,
        prompts: List[str],
        response_format=None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        max_tokens: int = 2048,
        desc: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> List[str]:
        """Process prompts concurrently via the external vLLM server.

        Uses ThreadPoolExecutor — the vLLM server handles continuous batching
        internally.

        Args:
            response_format: either an OpenAI-style response_format dict, e.g.
                ``{'type': 'json_schema', 'json_schema': {...}}``, or a class
                with a ``json_schema()`` classmethod (e.g. AgentResponse).
            system_prompt: system message; pass ``None`` for self-contained
                prompts (episode/rollout paths).
            enable_thinking: accepted for call-site compatibility — thinking is
                set at construction time on this backend; a mismatch is logged
                rather than silently dropped.

        Returns:
            List of response strings (thinking excluded, stripped), one per
            prompt, in input order. Exhausted retries yield "".
        """

        if not prompts:
            return []

        if enable_thinking is not None and enable_thinking != self.enable_thinking:
            logger.warning(
                f"generate_batch: enable_thinking={enable_thinking} requested but "
                f"server model was constructed with enable_thinking={self.enable_thinking}; "
                "construction-time setting wins."
            )
        if response_format is not None and hasattr(response_format, 'json_schema'):
            response_format = {
                'type': 'json_schema',
                'json_schema': {'name': 'structured_output',
                                'schema': response_format.json_schema()},
            }

        extra_body = {}
        if self.enable_thinking:
            extra_body['chat_template_kwargs'] = {"enable_thinking": True}

        max_retries = 3

        def _call_single(idx_prompt):
            idx, prompt = idx_prompt
            msgs = build_chat_messages(prompt, system_prompt)

            for attempt in range(max_retries):
                try:
                    create_kwargs = dict(
                        model=self.model_name,
                        messages=msgs,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if response_format is not None:
                        create_kwargs['response_format'] = response_format
                    if extra_body:
                        create_kwargs['extra_body'] = extra_body
                    resp = self.client.chat.completions.create(**create_kwargs)
                    msg = resp.choices[0].message
                    response_text = (msg.content or "").strip()
                    usage = getattr(resp, "usage", None)
                    self._record_trace(label, getattr(usage, "completion_tokens", None), True)

                    # Warn when structured output was requested but is unparseable
                    if response_format is not None and parse_json_response(response_text) is None:
                        logger.warning(
                            f"Prompt {idx}: structured output requested but JSON parse failed "
                            f"(likely truncated at max_tokens={max_tokens}). "
                            f"Returning raw text for caller to handle."
                        )
                    return idx, response_text
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Prompt {idx} attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                    else:
                        logger.error(f"Prompt {idx} failed after {max_retries} attempts: {e}")

            # All retries exhausted
            self._record_trace(label, None, False)
            return idx, ""

        label = desc or f"batch of {len(prompts)} prompts"
        logger.info(f"[vLLM Server] Generating {label} ...")

        results: List[Optional[str]] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            futures = {executor.submit(_call_single, (i, p)): i for i, p in enumerate(prompts)}
            for future in tqdm(as_completed(futures), total=len(prompts), desc=label):
                idx, text = future.result()
                results[idx] = text

        logger.info(f"Batch generation complete: {len(results)} responses")
        return results


class GPT:
    """OpenAI / Azure OpenAI API wrapper.

    The provider must be chosen explicitly — there is no environment-based
    fallback (the old ``USE_OPENAI_DIRECT`` env sniffing silently selected
    Azure OpenAI by default and was removed).
    """

    def __init__(self, model_name: str = "gpt-4o", max_workers: int = 8,
                 provider: str = "openai"):
        self.model_name = model_name
        self.max_workers = max_workers
        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "azure-openai":
            self.client = AzureOpenAI()
        else:
            raise ValueError(f"Unknown GPT provider {provider!r}; "
                             "expected 'openai' or 'azure-openai'")
        self.provider = provider

    def generate(self, prompt: str, temperature: float = 0.9, top_p: float = 0.95,
                 max_tokens: int = 2048, **kwargs) -> str:
        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e)
                if "content_filter" in err_str or "ResponsibleAIPolicy" in err_str:
                    logger.warning(f"GPT content filter triggered, skipping prompt: {e}")
                    return ""
                # Rate limit: parse server-suggested wait, or exponential backoff capped at 60s.
                if "rate_limit_exceeded" in err_str or "Rate limit" in err_str:
                    wait = None
                    m = re.search(r"try again in\s+([\d.]+)\s*(ms|s)", err_str, re.IGNORECASE)
                    if m:
                        val = float(m.group(1))
                        wait = val / 1000.0 if m.group(2).lower() == "ms" else val
                    if wait is None:
                        wait = min(60.0, 2 ** attempt)
                    wait += 1.0  # small safety pad
                    logger.warning(
                        f"GPT rate-limited (attempt {attempt + 1}/{max_attempts}), sleeping {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                logger.warning(f"GPT attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(min(30, 2 ** attempt))
                else:
                    raise

    def generate_batch(self, prompts: list, temperature: float = 0.9, top_p: float = 0.95,
                       max_tokens: int = 2048, desc: Optional[str] = None,
                       system_prompt: Optional[str] = None, response_format=None,
                       enable_thinking: Optional[bool] = None) -> list:
        # This backend has always sent a single user message and plain-text
        # completions; system_prompt / response_format / enable_thinking were
        # previously dropped silently by **kwargs — warn instead.
        for name, val in (("system_prompt", system_prompt),
                          ("response_format", response_format),
                          ("enable_thinking", enable_thinking)):
            if val:  # warn only on meaningful ignored values (False/None are no-ops)
                logger.warning(f"GPT.generate_batch ignores {name}={val!r} (unsupported)")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=min(len(prompts), self.max_workers)) as executor:
            futures = {
                executor.submit(self.generate, p, temperature, top_p, max_tokens): i
                for i, p in enumerate(prompts)
            }
            iterator = as_completed(futures)
            if desc:
                iterator = tqdm(iterator, total=len(futures), desc=desc)
            for future in iterator:
                results[futures[future]] = future.result()
        return results



class Claude:
    """Anthropic API wrapper."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        import anthropic

        self.model_name = model_name
        self.client = anthropic.Anthropic()

    def generate(self, prompt: str, temperature: float = 0.9, top_p: float = 0.95,
                 max_tokens: int = 2048, **kwargs) -> str:
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"Claude attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def generate_batch(self, prompts: list, temperature: float = 0.9, top_p: float = 0.95,
                       max_tokens: int = 2048, desc: Optional[str] = None,
                       system_prompt: Optional[str] = None, response_format=None,
                       enable_thinking: Optional[bool] = None) -> list:
        for name, val in (("system_prompt", system_prompt),
                          ("response_format", response_format),
                          ("enable_thinking", enable_thinking)):
            if val:  # warn only on meaningful ignored values (False/None are no-ops)
                logger.warning(f"Claude.generate_batch ignores {name}={val!r} (unsupported)")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=min(len(prompts), 8)) as executor:
            futures = {
                executor.submit(self.generate, p, temperature, top_p, max_tokens): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results



class Gemini:
    """Google Generative AI wrapper."""

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        import google.generativeai as genai

        self.model_name = model_name
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, temperature: float = 0.9, top_p: float = 0.95,
                 max_tokens: int = 2048, **kwargs) -> str:
        import google.generativeai as genai

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        for attempt in range(3):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                return response.text
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise


class Qwen:
    """HuggingFace Transformers wrapper (non-vLLM fallback)."""

    def __init__(self, model_name: str = DEFAULT_SPEAKER_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, prompt: str, temperature: float = 0.9, top_p: float = 0.95,
                 max_tokens: int = 2048, **kwargs) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )
        # Decode only the newly generated tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_batch(self, prompts: List[str], temperature: float = 0.9, top_p: float = 0.95,
                       max_tokens: int = 2048, desc: str = "Generating",
                       batch_size: int = 4, system_prompt: Optional[str] = None,
                       response_format=None, enable_thinking: Optional[bool] = None) -> List[str]:
        # HF fallback backend: raw-completion prompts; structured output and a
        # chat system prompt are not supported here — warn instead of dropping.
        for name, val in (("system_prompt", system_prompt),
                          ("response_format", response_format),
                          ("enable_thinking", enable_thinking)):
            if val:  # warn only on meaningful ignored values (False/None are no-ops)
                logger.warning(f"Qwen.generate_batch ignores {name}={val!r} (unsupported)")
        import torch

        results = []
        with tqdm(total=len(prompts), desc=desc) as pbar:
            for start in range(0, len(prompts), batch_size):
                chunk = prompts[start:start + batch_size]
                inputs = self.tokenizer(
                    chunk, return_tensors="pt", padding=True, truncation=True,
                ).to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                    )
                input_len = inputs["input_ids"].shape[1]
                for output in outputs:
                    generated = output[input_len:]
                    results.append(self.tokenizer.decode(generated, skip_special_tokens=True))
                pbar.update(len(chunk))
        return results



def get_llm(model_type: str, model_name: str, max_tokens: int = 2048,
            temperature: float = 0.9, top_p: float = 0.95) -> Any:
    """Factory function to create an LLM instance based on model_type."""
    model_type_lower = model_type.lower()
    if model_type_lower == "gpt":
        raise ValueError("model_type 'gpt' is ambiguous; use 'openai' or 'azure-openai'")
    if model_type_lower == "openai":
        return GPT(model_name=model_name, provider="openai")
    elif model_type_lower == "azure-openai":
        return GPT(model_name=model_name, provider="azure-openai")
    elif model_type_lower in ("claude", "anthropic"):
        return Claude(model_name=model_name)
    elif model_type_lower in ("gemini", "google"):
        return Gemini(model_name=model_name)
    elif model_type_lower in ("qwen", "transformers", "hf"):
        return Qwen(model_name=model_name)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. "
                         f"Supported: gpt, claude, gemini, qwen")


def load_vllm_model(
    model_name_or_path: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int = 1,
    enable_lora: bool = False,
    lora_path: Optional[str] = None,
    max_lora_rank: int = 64
) -> VLLMModel:
    """Load vLLM model with optional LoRA adapter support.

    Args:
        model_name_or_path: Model name (e.g., "Qwen/Qwen3-8B") or path to checkpoint
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        enable_lora: Whether to enable LoRA adapter support
        lora_path: Path to LoRA adapters (required if enable_lora=True)
        max_lora_rank: Maximum LoRA rank (default: 64)

    Returns:
        Initialized VLLMModel instance
    """
    logger.info(f"Loading model: {model_name_or_path}")
    logger.info(f"  GPU memory utilization: {gpu_memory_utilization}")

    model_kwargs = {
        'model_name': model_name_or_path,
        'tensor_parallel_size': tensor_parallel_size,
        'gpu_memory_utilization': gpu_memory_utilization,
    }

    if enable_lora:
        if not lora_path:
            raise ValueError("lora_path must be provided when enable_lora=True")
        logger.info(f"  LoRA enabled: {lora_path}")
        logger.info(f"  Max LoRA rank: {max_lora_rank}")
        model_kwargs['enable_lora'] = True
        model_kwargs['lora_path'] = lora_path
        model_kwargs['max_lora_rank'] = max_lora_rank

    model = VLLMModel(**model_kwargs)
    logger.info("  Model loaded successfully")
    return model


# Roles that fall back to the shared --vllm-server-url. The judge endpoint is
# independently configurable and never inherits the shared URL.
SHARED_URL_ROLES = ("director", "speaker", "user")


def resolve_role_identity(clients: Dict[str, Any],
                          requested: Dict[str, str]) -> Dict[str, Dict[str, Optional[str]]]:
    """Base-model vs adapter identity per role, from live client preflight data.

    For a served LoRA alias the client knows its base (``lora_parent``) and the
    alias id; for base models / local clients the adapter is None. This is
    runtime evidence (from /v1/models), not configuration.
    """
    identity = {}
    for role, client in clients.items():
        req = requested.get(role)
        if getattr(client, "is_lora_alias", False):
            identity[role] = {"base_model": client.lora_parent,
                              "adapter": client.model_name}
        else:
            identity[role] = {"base_model": getattr(client, "model_name", req),
                              "adapter": None}
    return identity


def get_client(
    role: str,
    *,
    model_name: str,
    server_urls: Optional[Dict[str, Optional[str]]] = None,
    enable_thinking: bool = True,
    gpu_memory_utilization: float = 0.7,
    tensor_parallel_size: int = 1,
    lora_path: Optional[str] = None,
    max_concurrent_requests: int = 32,
    trace: Optional[list] = None,
) -> Union[VLLMModel, "VLLMServerModel"]:
    """Single factory for role-specific vLLM clients (server or local).

    ``server_urls`` maps roles ('director', 'speaker', 'user', 'judge') to
    URLs plus an optional 'shared' entry (--vllm-server-url). Resolution:
    role-specific URL > shared URL (director/speaker/user only) > local
    in-process model. The judge role never falls back to the shared URL.
    ``trace``: optional shared list receiving per-request role-call records
    (server backend only).
    """
    server_urls = server_urls or {}
    url = server_urls.get(role)
    if url is None and role in SHARED_URL_ROLES:
        url = server_urls.get("shared")

    if url:
        if lora_path:
            raise ValueError(
                f"Role '{role}': a local LoRA path ({lora_path}) cannot be applied "
                "to an external vLLM server. Serve the adapter on the server "
                "(scripts/serve_vllm.py --lora-modules alias=path) or merge it "
                "first (src.train.merge_lora), or drop the server URL to load "
                "the model locally.")
        logger.info(f"[{role}] connecting to vLLM server at {url} (model: {model_name})")
        return VLLMServerModel(
            model_name=model_name,
            base_url=url,
            enable_thinking=enable_thinking,
            max_concurrent_requests=max_concurrent_requests,
            role=role,
            trace=trace,
        )

    if trace is not None:
        logger.warning(f"[{role}] role-call tracing is only supported on the "
                       "server backend; local model calls will not be traced")
    logger.info(f"[{role}] loading local vLLM model {model_name}")
    return load_vllm_model(
        model_name_or_path=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enable_lora=lora_path is not None,
        lora_path=lora_path,
    )


def get_shared_clients(role_specs: Dict[str, Dict], server_urls: Dict[str, Optional[str]],
                       *,
                       enable_thinking: bool = True,
                       share_model: bool = False,
                       max_concurrent_requests: int = 32,
                       trace: Optional[list] = None) -> Dict[str, Any]:
    """Build clients for several roles.

    ``role_specs`` maps role → {model_name, lora_path?, gpu_memory_utilization?,
    tensor_parallel_size?}.

    Sharing semantics:
    - ``share_model=True`` forces the director spec to the speaker spec (the
      historical --share-model behavior); the user role is never forced.
    - Server (HTTP) clients are reused freely: identical
      (url, model_name, thinking) → one client.
    - LOCAL models are reused ONLY under explicit ``share_model=True``;
      otherwise identical local specs are loaded separately (2× weight VRAM —
      the per-role gpu_memory_utilization fractions must sum to < 1.0 per GPU).
    """
    role_specs = {role: dict(spec) for role, spec in role_specs.items()}
    if share_model and "director" in role_specs and "speaker" in role_specs:
        logger.info("--share-model: director reuses the speaker model spec")
        role_specs["director"] = dict(role_specs["speaker"])

    clients: Dict[str, Any] = {}
    cache: Dict[tuple, Any] = {}
    seen_local_keys: Dict[tuple, str] = {}
    for role, spec in role_specs.items():
        url = server_urls.get(role)
        if url is None and role in SHARED_URL_ROLES:
            url = server_urls.get("shared")
        if url:
            key = ("server", url, spec["model_name"], enable_thinking)
        else:
            key = ("local", spec["model_name"], spec.get("lora_path"),
                   spec.get("tensor_parallel_size", 1),
                   spec.get("gpu_memory_utilization", 0.7), enable_thinking)
            if not share_model:
                # Local models are shared only on explicit request.
                if key in seen_local_keys:
                    logger.warning(
                        f"Roles {seen_local_keys[key]!r} and {role!r} have identical "
                        f"local specs; loading {spec['model_name']} TWICE (2x weight "
                        "VRAM; per-role gpu_memory_utilization fractions must sum to "
                        "<1.0 per GPU). Pass --share-model to reuse one instance.")
                seen_local_keys.setdefault(key, role)
                key = key + (role,)
        if key not in cache:
            cache[key] = get_client(
                role,
                model_name=spec["model_name"],
                server_urls=server_urls,
                enable_thinking=enable_thinking,
                gpu_memory_utilization=spec.get("gpu_memory_utilization", 0.7),
                tensor_parallel_size=spec.get("tensor_parallel_size", 1),
                lora_path=spec.get("lora_path"),
                max_concurrent_requests=max_concurrent_requests,
                trace=trace,
            )
        else:
            logger.info(f"[{role}] reusing {key[0]} client for {spec['model_name']}")
        clients[role] = cache[key]
    return clients
