"""Automatic (non-LLM) metric evaluators: NLI contradiction detection & Self-BLEU.

Separate from the LLM rubric evaluators because these carry their own
dependencies (transformers NLI model, NLTK BLEU) — install the ``metrics``
extra to use them.
"""

import logging
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)



def _clean_response(text: str) -> str:
    """Remove <think></think> tags from a response."""
    if "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text.strip()


def _detect_format(conversation_data: Dict) -> str:
    """Auto-detect the data format of a conversation record.

    Returns one of: 'episodes', 'turns', 'agent_conversations', 'segments'.
    Raises ValueError if no known format is detected.
    """
    if "episodes" in conversation_data:
        return "episodes"
    if "turns" in conversation_data and isinstance(
        conversation_data["turns"], list
    ):
        # Flat turns list (MASCOT eval output)
        return "turns"
    if "agent_conversations" in conversation_data:
        return "agent_conversations"
    if "segments" in conversation_data:
        return "segments"
    raise ValueError(
        f"Cannot detect data format. Available keys: "
        f"{list(conversation_data.keys())}"
    )


class NLIContradictionEvaluator:
    """Evaluates inter-agent coherence using NLI-based contradiction detection.

    Uses roberta-large-mnli to classify all agent response pairs within an
    episode/round as entailment, neutral, or contradiction. High contradiction
    rates indicate agents are producing incoherent or conflicting statements.
    """

    NLI_LABELS = ["contradiction", "neutral", "entailment"]

    def __init__(self, device: str = "cuda", batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the NLI model on first use."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = "roberta-large-mnli"
        logger.info(f"Loading NLI model: {model_name} on {self.device}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        if self.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.half().to("cuda")
        else:
            self.device = "cpu"
            self._model = self._model.to("cpu")
        self._model.eval()
        logger.info("NLI model loaded successfully")

    def _classify_pairs(self, pairs: List[Tuple[str, str]]) -> List[str]:
        """Run NLI classification on a list of (premise, hypothesis) pairs."""
        import torch
        self._load_model()
        labels = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]
            inputs = self._tokenizer(
                premises, hypotheses,
                padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            labels.extend(self.NLI_LABELS[p] for p in preds)
        return labels

    def evaluate_conversation(
        self, conversation_data: Dict, agents: Dict, dataset_name: str
    ) -> Dict[str, Any]:
        """Evaluate a single conversation for inter-agent contradictions.

        Args:
            conversation_data: One conversation record from JSONL.
            agents: Agent config dict (AGENTS or EMPATHETIC_AGENTS).
            dataset_name: 'empathetic' or 'qmsum'.

        Returns:
            Dict with contradiction_rate, entailment_rate, neutral_rate,
            total_pairs, and per_pair breakdown.
        """
        pairs, pair_labels_meta = self._extract_pairs(
            conversation_data, agents, dataset_name
        )
        if not pairs:
            return {
                "contradiction_rate": 0.0, "entailment_rate": 0.0,
                "neutral_rate": 0.0, "total_pairs": 0, "pair_details": [],
            }

        labels = self._classify_pairs(pairs)

        counts = {"contradiction": 0, "neutral": 0, "entailment": 0}
        pair_details = []
        for label, meta_item in zip(labels, pair_labels_meta):
            counts[label] += 1
            pair_details.append({**meta_item, "nli_label": label})

        total = len(labels)
        return {
            "contradiction_rate": counts["contradiction"] / total,
            "entailment_rate": counts["entailment"] / total,
            "neutral_rate": counts["neutral"] / total,
            "total_pairs": total,
            "pair_details": pair_details,
        }

    def evaluate_batch(
        self, conversations: List[Dict], agents: Dict, dataset_name: str
    ) -> List[Dict[str, Any]]:
        """Evaluate all conversations in one batched GPU pass.

        Same output format as evaluate_conversation(), one dict per conversation.
        """
        all_pairs: List[Tuple[str, str]] = []
        all_meta: List[List[Dict]] = []
        conv_pair_counts: List[int] = []

        for conv in conversations:
            pairs, meta = self._extract_pairs(conv, agents, dataset_name)
            conv_pair_counts.append(len(pairs))
            all_pairs.extend(pairs)
            all_meta.append(meta)

        # Single batched GPU inference over all pairs
        if all_pairs:
            all_labels = self._classify_pairs(all_pairs)
        else:
            all_labels = []

        # Slice labels back into per-conversation chunks
        results: List[Dict[str, Any]] = []
        offset = 0
        for idx, count in enumerate(conv_pair_counts):
            if count == 0:
                results.append({
                    "contradiction_rate": 0.0, "entailment_rate": 0.0,
                    "neutral_rate": 0.0, "total_pairs": 0, "pair_details": [],
                })
            else:
                chunk_labels = all_labels[offset : offset + count]
                meta = all_meta[idx]
                counts = {"contradiction": 0, "neutral": 0, "entailment": 0}
                pair_details = []
                for label, meta_item in zip(chunk_labels, meta):
                    counts[label] += 1
                    pair_details.append({**meta_item, "nli_label": label})
                total = len(chunk_labels)
                results.append({
                    "contradiction_rate": counts["contradiction"] / total,
                    "entailment_rate": counts["entailment"] / total,
                    "neutral_rate": counts["neutral"] / total,
                    "total_pairs": total,
                    "pair_details": pair_details,
                })
            offset += count

        return results

    def _extract_pairs(
        self, conversation_data: Dict, agents: Dict, dataset_name: str
    ) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        """Extract all agent-response pairs for NLI evaluation."""
        pairs: List[Tuple[str, str]] = []
        meta: List[Dict] = []
        fmt = _detect_format(conversation_data)

        if fmt == "episodes":
            for ep_idx, episode in enumerate(
                conversation_data.get("episodes", [])
            ):
                agent_responses = []
                for turn in episode.get("turns", []):
                    if turn.get("turn_type") == "agent":
                        agent_responses.append(
                            (turn["agent_name"],
                             _clean_response(turn["response"]))
                        )
                for (a1, r1), (a2, r2) in combinations(agent_responses, 2):
                    pairs.append((r1, r2))
                    meta.append({
                        "episode": ep_idx, "agent_a": a1, "agent_b": a2,
                    })
        elif fmt == "turns":
            agent_responses = []
            for turn in conversation_data.get("turns", []):
                if turn.get("turn_type") == "agent":
                    agent_responses.append(
                        (turn["agent_name"],
                         _clean_response(turn["response"]))
                    )
            for (a1, r1), (a2, r2) in combinations(agent_responses, 2):
                pairs.append((r1, r2))
                meta.append({"agent_a": a1, "agent_b": a2})
        elif fmt == "segments":
            for seg_idx, segment in enumerate(
                conversation_data.get("segments", [])
            ):
                agent_responses = []
                for r in segment.get("responses", []):
                    text = _clean_response(r.get("response", ""))
                    if text:
                        agent_responses.append((r["agent_key"], text))
                for (a1, r1), (a2, r2) in combinations(agent_responses, 2):
                    pairs.append((r1, r2))
                    meta.append({
                        "segment": seg_idx, "agent_a": a1, "agent_b": a2,
                    })
        else:  # agent_conversations
            agent_convs = conversation_data.get("agent_conversations", {})
            agent_names = list(agent_convs.keys())
            if not agent_names:
                return [], []
            num_rounds = len(agent_convs[agent_names[0]])
            for round_idx in range(num_rounds):
                round_responses = []
                for aname in agent_names:
                    responses = agent_convs[aname]
                    if round_idx < len(responses):
                        items = responses[round_idx]
                        if isinstance(items, list):
                            text = _clean_response(items[-1]) if items else ""
                        else:
                            text = _clean_response(str(items))
                        if text:
                            round_responses.append((aname, text))
                for (a1, r1), (a2, r2) in combinations(round_responses, 2):
                    pairs.append((r1, r2))
                    meta.append({
                        "round": round_idx, "agent_a": a1, "agent_b": a2,
                    })

        return pairs, meta


class SelfBLEUEvaluator:
    """Evaluates response diversity across agents using Self-BLEU.

    For each agent response, calculates BLEU against all other agent responses
    in the same episode/round. Lower Self-BLEU indicates higher diversity
    (desirable for multi-agent systems).
    """

    def __init__(self, max_ngram: int = 4):
        self.max_ngram = max_ngram

    def _compute_self_bleu(
        self, responses: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """Compute Self-BLEU for a set of (agent_name, response) pairs."""
        import numpy as np
        from nltk.translate.bleu_score import (
            sentence_bleu, SmoothingFunction,
        )

        if len(responses) < 2:
            return {"mean_self_bleu": 0.0, "scores": []}

        smoothing = SmoothingFunction().method1
        weights = tuple(1.0 / self.max_ngram for _ in range(self.max_ngram))
        tokenized = [(name, resp.split()) for name, resp in responses]
        scores = []
        for i, (name_i, toks_i) in enumerate(tokenized):
            refs = [toks_j for j, (_, toks_j) in enumerate(tokenized) if j != i]
            bleu = sentence_bleu(
                refs, toks_i, weights=weights,
                smoothing_function=smoothing,
            )
            scores.append({"agent": name_i, "self_bleu": bleu})

        return {
            "mean_self_bleu": float(
                np.mean([s["self_bleu"] for s in scores])
            ),
            "scores": scores,
        }

    def evaluate_conversation(
        self, conversation_data: Dict, agents: Dict, dataset_name: str
    ) -> Dict[str, Any]:
        """Evaluate a single conversation for response diversity.

        Args:
            conversation_data: One conversation record from JSONL.
            agents: Agent config dict (AGENTS or EMPATHETIC_AGENTS).
            dataset_name: 'empathetic' or 'qmsum'.

        Returns:
            Dict with mean_self_bleu, std_self_bleu, and per_agent_self_bleu.
        """
        import numpy as np

        episode_scores = []
        per_agent: Dict[str, List[float]] = {}
        fmt = _detect_format(conversation_data)

        if fmt == "episodes":
            for episode in conversation_data.get("episodes", []):
                agent_responses = []
                for turn in episode.get("turns", []):
                    if turn.get("turn_type") == "agent":
                        agent_responses.append(
                            (turn["agent_name"],
                             _clean_response(turn["response"]))
                        )
                if len(agent_responses) < 2:
                    continue
                result = self._compute_self_bleu(agent_responses)
                episode_scores.append(result["mean_self_bleu"])
                for s in result["scores"]:
                    per_agent.setdefault(s["agent"], []).append(
                        s["self_bleu"]
                    )
        elif fmt == "turns":
            # Flat turns list — treat all agent turns as one group
            agent_responses = []
            for turn in conversation_data.get("turns", []):
                if turn.get("turn_type") == "agent":
                    agent_responses.append(
                        (turn["agent_name"],
                         _clean_response(turn["response"]))
                    )
            if len(agent_responses) >= 2:
                result = self._compute_self_bleu(agent_responses)
                episode_scores.append(result["mean_self_bleu"])
                for s in result["scores"]:
                    per_agent.setdefault(s["agent"], []).append(
                        s["self_bleu"]
                    )
        elif fmt == "segments":
            for segment in conversation_data.get("segments", []):
                agent_responses = []
                for r in segment.get("responses", []):
                    text = _clean_response(r.get("response", ""))
                    if text:
                        agent_responses.append((r["agent_key"], text))
                if len(agent_responses) < 2:
                    continue
                result = self._compute_self_bleu(agent_responses)
                episode_scores.append(result["mean_self_bleu"])
                for s in result["scores"]:
                    per_agent.setdefault(s["agent"], []).append(
                        s["self_bleu"]
                    )
        else:  # agent_conversations
            agent_convs = conversation_data.get("agent_conversations", {})
            agent_names = list(agent_convs.keys())
            if not agent_names:
                return {
                    "mean_self_bleu": 0.0, "std_self_bleu": 0.0,
                    "per_agent_self_bleu": {},
                }
            num_rounds = len(agent_convs[agent_names[0]])
            for round_idx in range(num_rounds):
                round_responses = []
                for aname in agent_names:
                    responses = agent_convs[aname]
                    if round_idx < len(responses):
                        items = responses[round_idx]
                        if isinstance(items, list):
                            text = _clean_response(items[-1]) if items else ""
                        else:
                            text = _clean_response(str(items))
                        if text:
                            round_responses.append((aname, text))
                if len(round_responses) < 2:
                    continue
                result = self._compute_self_bleu(round_responses)
                episode_scores.append(result["mean_self_bleu"])
                for s in result["scores"]:
                    per_agent.setdefault(s["agent"], []).append(
                        s["self_bleu"]
                    )

        if not episode_scores:
            return {
                "mean_self_bleu": 0.0, "std_self_bleu": 0.0,
                "per_agent_self_bleu": {},
            }

        return {
            "mean_self_bleu": float(np.mean(episode_scores)),
            "std_self_bleu": float(np.std(episode_scores)),
            "per_agent_self_bleu": {
                k: float(np.mean(v)) for k, v in per_agent.items()
            },
        }
