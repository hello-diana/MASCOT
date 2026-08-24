"""Individual and paired candidate-generation pipelines.

Moved verbatim from src/dataset/generate_and_annotate.py (Phase-2 Part A);
generation parameters, prompts, parsing, and pair semantics are unchanged.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

from config.model_config import ModelConfig
from config.persona_config import AGENTS, EMPATHETIC_AGENTS
from src.const import (DATASET2HF, EMPATHETIC_DATASET, ESCONV_DATASET,
                       PAIRED, QMSUM_DATASET)
from src.data_generation.annotation import (annotate_individual_queries,
                                            verify_pairs_with_llm)
from src.data_generation.common import artifact_paths, clean, scale_scores
from src.data_generation.preference_pairs import (build_individual_pairs,
                                                  build_paired_pairs)
from src.data_generation.serialization import (read_jsonl, save_pairs_dataset,
                                               write_jsonl)
from src.eval.rubric_evaluator import IndividualRubricEvaluator
from src.models.llm_models import VLLMServerModel
from src.prompts.persona_templates import MascotPersonaTemplates, get_summary_prompt
from src.utils.data_utils import (load_empathetic_situations,
                                  load_esconv_situations)
from src.utils.text import strip_response_prefix

logger = logging.getLogger(__name__)


def _build_paired_prompt_empathetic(persona, situation: str, rubric_text: str) -> str:
    """Build a rubric-aware prompt that asks for one good + one somewhat-worse response.

    The worse response should target rubric score ≈3 (generic, predictable but on-topic,
    coherent, persona-consistent), NOT a trivially-bad score-1/2 response.
    """
    traits_str = "\n".join(f"- {t}" for t in persona.traits)
    return f"""You are {persona.name}, an emotionally supportive agent.

## Persona Bio
{persona.description_long or persona.name}

## Key traits you must embody
{traits_str}

## Evaluation Rubric (this will be used to score both of your responses)
{rubric_text}

## Your Task: Generate ONE PAIR of responses for the situation above
You must produce exactly TWO responses to the same user situation:

1. "good_response" — TARGET RUBRIC SCORE ≈ 4
   - Vivid persona voice with character-specific vocabulary
   - Names a SPECIFIC unfelt emotion or offers non-obvious insight / reframe
   - Goes beyond surface relevance to deeper unstated implications
   - Introduces a genuinely NEW angle the user hasn't considered

2. "worse_response" — TARGET RUBRIC SCORE ≈ 3 (NOT lower)
   - MUST remain on-topic, coherent, and persona-consistent.
   - MUST be non-trivially bad.
   - MUST NOT be tone-deaf, dismissive, off-topic, or contradictory
   - Uses STANDARD coaching/therapy language (e.g., "What do you think...", "Tell me more...", "I'm here with you...")
   - Functional but generic — recognizable persona but predictable, lacks specific insight
   - The difference vs. good_response should be DEPTH and SPECIFICITY, not character or coherence

## Output Format (STRICT — JSON only, no prose, no markdown fences)
{{
    "good_response": "...",
    "worse_response": "..."
}}

Both responses must be in English, 1-3 sentences, and address the SAME user situation.

## Respond to the following user situation
{situation}

"""


def _parse_paired_response(raw_text: str) -> Optional[Tuple[str, str]]:
    """Extract (good_response, worse_response) from LLM output. Returns None on failure."""
    cleaned = clean(raw_text)
    if not cleaned:
        return None
    m = re.search(r'\{[\s\S]*\}', cleaned)
    if not m:
        logger.warning(f"Paired parse: no JSON object found in output (first 200 chars): {cleaned[:200]!r}")
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"Paired parse: JSON decode failed ({e}); snippet: {m.group(0)[:200]!r}")
        return None
    good = (parsed.get("good_response") or "").strip()
    worse = (parsed.get("worse_response") or "").strip()
    if not good or not worse:
        logger.warning(f"Paired parse: missing good/worse fields in {list(parsed.keys())}")
        return None
    return good, worse


def individual_conversation_pipeline(args, llm: VLLMServerModel, judge_llm: VLLMServerModel):
    """Generate individual responses per query (N independent OR one good+worse pair), score, form preference pairs."""
    if args.mode == PAIRED and args.dataset == QMSUM_DATASET:
        raise ValueError("--mode paired only supports Empathetic / ESConv (rubric is empathy-centric)")

    agents = EMPATHETIC_AGENTS if args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET) else AGENTS
    config = ModelConfig()
    config.base_model = args.model_name
    N = args.num_candidates  # K candidate responses per query
    id_key = "meeting_id" if args.dataset == QMSUM_DATASET else "conv_id"

    # ---- Phase 1: prepare queries ----
    queries: List[dict] = []  # each: {id, prompt_text, persona, ...}

    if args.dataset == QMSUM_DATASET:
        dataset = load_dataset(DATASET2HF[QMSUM_DATASET], split="train")
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        # Pre-generate summaries
        meetings = []
        for item in dataset:
            transcript = [{"speaker": u["speaker"], "text": u["content"]}
                          for u in item.get("meeting_transcripts", [])]
            meetings.append({"meeting_id": item["id"], "transcript": transcript})

        logger.info(f"Generating summaries for {len(meetings)} meetings...")
        summary_prompts = [
            get_summary_prompt("".join(f"{t['speaker']}: {t['text']}\n"
                                       for t in m["transcript"][:400]))
            for m in meetings
        ]
        summaries = [clean((t or "").strip()) for t in llm.generate_batch(
            prompts=summary_prompts, temperature=0.7,
            max_tokens=config.max_new_tokens,
            desc=f"Summaries ({len(summary_prompts)})",
        )]

        # Build per-agent queries (one per meeting × agent)
        TURNS_PER_ROUND = args.turns_per_round
        for mi, m in enumerate(meetings):
            chunks = []
            for i in range(0, len(m["transcript"]), TURNS_PER_ROUND):
                chunks.append(m["transcript"][i:i + TURNS_PER_ROUND])
            accumulated = ""
            for ci, chunk in enumerate(chunks):
                chunk_text = "".join(f"{t['speaker']}: {t['text']}\n" for t in chunk)
                ctx = (accumulated + "\n" + chunk_text).strip() if accumulated else chunk_text
                for aname, pspec in agents.items():
                    prompt_text = MascotPersonaTemplates.speaker_prompt(
                        QMSUM_DATASET, pspec, ctx, summaries[mi])
                    queries.append({
                        "id": m["meeting_id"],
                        "prompt_text": prompt_text,
                        "persona": aname,
                        "context": ctx,
                    })
                accumulated = ctx

    elif args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET):
        if args.dataset == ESCONV_DATASET:
            conv_list = load_esconv_situations(split="train", max_samples=args.max_samples)
        else:
            conv_list = load_empathetic_situations(split="train", max_samples=args.max_samples)
        for c in conv_list:
            c["situation"] = c["situation"].replace("__comma__", ", ")

        for conv in conv_list:
            for aname, pspec in agents.items():
                if args.mode == PAIRED:
                    prompt_text = _build_paired_prompt_empathetic(
                        persona=pspec, situation=conv["situation"],
                        rubric_text=IndividualRubricEvaluator.RUBRIC,
                    )
                else:
                    prompt_text = MascotPersonaTemplates.speaker_prompt_empathetic(
                        persona=pspec, context=conv["situation"],
                        conversation_history="", num_shot=3,
                    )
                queries.append({
                    "id": conv["conv_id"],
                    "prompt_text": prompt_text,
                    "persona": aname,
                    "context": conv["situation"],
                })

    logger.info(f"Total queries: {len(queries)}")

    # ---- Phase 2: Generate responses per query (with replacement) ----
    _paths = artifact_paths(args)
    gen_file = _paths["raw_file"]
    os.makedirs(os.path.dirname(gen_file), exist_ok=True)

    if os.path.exists(gen_file):
        logger.info(f"Found existing generations at {gen_file}, skipping generation")
        queries = read_jsonl(gen_file)
    elif args.mode == PAIRED:
        K = args.num_pairs_per_query
        all_prompts = []
        prompt_idx_map = []  # (query_idx, pair_idx)
        for qi, q in enumerate(queries):
            for pi in range(K):
                all_prompts.append(q["prompt_text"])
                prompt_idx_map.append((qi, pi))

        logger.info(f"Paired generation: {len(all_prompts)} pair calls ({len(queries)} queries × {K})...")
        raw_outputs = [(t or "").strip() for t in llm.generate_batch(
            prompts=all_prompts, temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            desc=f"Paired candidates ({len(all_prompts)})",
        )]

        # Parse pairs; drop any query whose ANY pair fails to parse so q["responses"]
        # length stays consistent (2*K) for downstream phases.
        per_query_pairs: Dict[int, List[Optional[Tuple[str, str]]]] = {
            qi: [None] * K for qi in range(len(queries))
        }
        for idx, (qi, pi) in enumerate(prompt_idx_map):
            per_query_pairs[qi][pi] = _parse_paired_response(raw_outputs[idx])

        kept_queries = []
        n_dropped = 0
        for qi, q in enumerate(queries):
            pairs = per_query_pairs[qi]
            if any(p is None for p in pairs):
                n_dropped += 1
                continue
            # Flatten [good_0, worse_0, good_1, worse_1, ...]
            flat = []
            for good, worse in pairs:
                flat.extend([good, worse])
            q["responses"] = flat
            kept_queries.append(q)
        logger.info(f"Paired parse: kept {len(kept_queries)}/{len(queries)} queries "
                    f"(dropped {n_dropped} with parse failures)")
        queries = kept_queries

        write_jsonl(gen_file, queries)
        logger.info(f"Saved raw paired generations to {gen_file}")
    else:
        all_prompts = []
        prompt_idx_map = []  # (query_idx, response_idx)
        for qi, q in enumerate(queries):
            for ri in range(N):
                all_prompts.append(q["prompt_text"])
                prompt_idx_map.append((qi, ri))

        logger.info(f"Generating {len(all_prompts)} candidate responses...")
        all_responses = [strip_response_prefix((t or "").strip()) for t in llm.generate_batch(
            prompts=all_prompts, temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            desc=f"Individual candidates ({len(all_prompts)})",
        )]

        # Attach responses to queries
        for q in queries:
            q["responses"] = [None] * N
        for idx, (qi, ri) in enumerate(prompt_idx_map):
            queries[qi]["responses"][ri] = all_responses[idx]

        # Save raw generation
        write_jsonl(gen_file, queries)
        logger.info(f"Saved raw generations to {gen_file}")

    # ---- Phase 3: Annotate (parallel) ----
    ds = EMPATHETIC_DATASET if args.dataset in (EMPATHETIC_DATASET, ESCONV_DATASET) else QMSUM_DATASET
    evaluator = IndividualRubricEvaluator(judge_llm, dataset_name=ds)
    annotate_individual_queries(queries, evaluator, agents, ds, args,
                                _paths["annotated_file"])

    # ---- Phase 4: Format preference pairs ----
    if args.mode == PAIRED:
        pairs = build_paired_pairs(queries, args, id_key)
    else:
        pairs = build_individual_pairs(queries, args, id_key)
    logger.info(f"Created {len(pairs)} preference pairs (before verification)")

    # ---- Phase 5: LLM verification ----
    if args.verify_pairs and pairs:
        pairs = verify_pairs_with_llm(pairs, judge_llm)

    # ---- Phase 6: Scale and save ----
    if not pairs:
        logger.warning("No preference pairs to save!")
        return

    pairs = scale_scores(pairs)
    return save_pairs_dataset(pairs, _paths["dataset_dir"], id_key=id_key)
