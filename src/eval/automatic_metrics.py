"""Automatic (non-LLM) metric evaluators: NLI contradiction detection & Self-BLEU.

Separate from the LLM rubric evaluators because these carry their own
dependencies (transformers NLI model, NLTK BLEU) — install the ``metrics``
extra to use them.

CLI::

    python -m src.eval.automatic_metrics \
        --input-file outputs/mascot_eval/.../mascot_*.jsonl \
        --output-file outputs/metrics.json \
        --pairing-mode same-round --nli-direction temporal

NLI direction:
- ``temporal`` (paper default): the earlier agent turn is the premise and the
  later turn the hypothesis — one classification per pair. This preserves the
  paper's metric definition.
- ``bidirectional`` (optional robustness analysis): classify both A→B and B→A
  and aggregate (contradiction if either direction contradicts; entailment iff
  both entail; else neutral).

Pairing modes:
- ``same-round`` (default): pair agent responses within the same round, using
  each turn's explicit ``round_id`` metadata. Legacy files without round_id
  FAIL loudly — pass ``--allow-round-inference`` to fall back to user-turn
  delimited round inference, or choose another pairing mode explicitly.
  (same-round is never silently degraded to all-pairs.)
- ``adjacent-agent``: consecutive agent turns only.
- ``all-pairs``: every agent-response combination in the conversation.
"""

import argparse
import json
import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "FacebookAI/roberta-large-mnli"
PAIRING_MODES = ("same-round", "adjacent-agent", "all-pairs")
NLI_DIRECTIONS = ("temporal", "bidirectional")


class MissingRoundIdError(ValueError):
    """same-round pairing was requested but the data has no round_id metadata."""



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

    def __init__(self, device: str = "cuda", batch_size: int = 32,
                 model_name: str = DEFAULT_NLI_MODEL,
                 direction: str = "temporal",
                 pairing_mode: str = "same-round",
                 allow_round_inference: bool = False):
        if direction not in NLI_DIRECTIONS:
            raise ValueError(f"direction must be one of {NLI_DIRECTIONS}, got {direction!r}")
        if pairing_mode not in PAIRING_MODES:
            raise ValueError(f"pairing_mode must be one of {PAIRING_MODES}, got {pairing_mode!r}")
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.direction = direction
        self.pairing_mode = pairing_mode
        self.allow_round_inference = allow_round_inference
        self._model = None
        self._tokenizer = None
        self._id2label = None

    def _load_model(self):
        """Lazy-load the NLI model on first use."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"Loading NLI model: {self.model_name} on {self.device}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        # Label order comes from the model config — never hardcode indices.
        self._id2label = {
            int(i): str(label).lower()
            for i, label in self._model.config.id2label.items()
        }
        logger.info(f"NLI labels from model config: {self._id2label}")

        if self.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.half().to("cuda")
        else:
            self.device = "cpu"
            self._model = self._model.to("cpu")
        self._model.eval()
        logger.info("NLI model loaded successfully")

    def _classify_directed(self, pairs: List[Tuple[str, str]]) -> List[str]:
        """Run one-directional NLI classification on (premise, hypothesis) pairs."""
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
            labels.extend(self._id2label[p] for p in preds)
        return labels

    @staticmethod
    def _aggregate_bidirectional(label_ab: str, label_ba: str) -> str:
        """Contradiction if either direction; entailment iff both; else neutral."""
        if "contradiction" in (label_ab, label_ba):
            return "contradiction"
        if label_ab == "entailment" and label_ba == "entailment":
            return "entailment"
        return "neutral"

    def _classify_pairs(self, pairs: List[Tuple[str, str]]):
        """Classify pairs per the configured direction.

        temporal: premise = earlier response, hypothesis = later response
        (pairs arrive in temporal order). Returns (labels, None).
        bidirectional: classify both directions and aggregate. Returns
        (labels, list of (label_ab, label_ba)).
        """
        if self.direction == "temporal":
            return self._classify_directed(pairs), None
        labels_ab = self._classify_directed(pairs)
        labels_ba = self._classify_directed([(h, p) for p, h in pairs])
        aggregated = [self._aggregate_bidirectional(ab, ba)
                      for ab, ba in zip(labels_ab, labels_ba)]
        return aggregated, list(zip(labels_ab, labels_ba))

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

        labels, directions = self._classify_pairs(pairs)

        counts = {"contradiction": 0, "neutral": 0, "entailment": 0}
        pair_details = []
        for i, (label, meta_item) in enumerate(zip(labels, pair_labels_meta)):
            counts[label] += 1
            detail = {**meta_item, "nli_label": label}
            if directions is not None:
                detail["label_ab"], detail["label_ba"] = directions[i]
            pair_details.append(detail)

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
            all_labels, all_directions = self._classify_pairs(all_pairs)
        else:
            all_labels, all_directions = [], None

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
                for j, (label, meta_item) in enumerate(zip(chunk_labels, meta)):
                    counts[label] += 1
                    detail = {**meta_item, "nli_label": label}
                    if all_directions is not None:
                        detail["label_ab"], detail["label_ba"] = all_directions[offset + j]
                    pair_details.append(detail)
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

    def _extract_groups(
        self, conversation_data: Dict
    ) -> List[Tuple[Dict, List[Tuple[str, str]]]]:
        """Collect agent responses as (group_meta, [(agent, text), ...]) groups
        in temporal order. A group is one round/episode/segment.
        """
        fmt = _detect_format(conversation_data)
        groups: List[Tuple[Dict, List[Tuple[str, str]]]] = []

        if fmt == "episodes":
            for ep_idx, episode in enumerate(conversation_data.get("episodes", [])):
                responses = [
                    (turn["agent_name"], _clean_response(turn["response"]))
                    for turn in episode.get("turns", [])
                    if turn.get("turn_type") == "agent"
                ]
                groups.append(({"episode": ep_idx}, responses))
        elif fmt == "turns":
            turns = [t for t in conversation_data.get("turns", [])
                     if t.get("turn_type") == "agent"]
            has_round_ids = turns and all("round_id" in t for t in turns)
            if has_round_ids:
                by_round: Dict[Any, List[Tuple[str, str]]] = {}
                order = []
                for t in turns:
                    rid = t["round_id"]
                    if rid not in by_round:
                        by_round[rid] = []
                        order.append(rid)
                    by_round[rid].append(
                        (t["agent_name"], _clean_response(t["response"])))
                groups = [({"round": rid}, by_round[rid]) for rid in order]
            elif self.pairing_mode == "same-round":
                if not self.allow_round_inference:
                    raise MissingRoundIdError(
                        "same-round pairing requires per-turn round_id metadata, "
                        "which this file does not carry (legacy output). Pass "
                        "--allow-round-inference to infer rounds from user turns, "
                        "or choose --pairing-mode all-pairs / adjacent-agent "
                        "explicitly. same-round is never silently degraded.")
                logger.warning(
                    "round_id missing; inferring rounds from user-turn boundaries "
                    "(--allow-round-inference). A conversation without user turns "
                    "degenerates to a single round.")
                current: List[Tuple[str, str]] = []
                rid = 0
                for t in conversation_data.get("turns", []):
                    if t.get("turn_type") == "agent":
                        current.append(
                            (t["agent_name"], _clean_response(t["response"])))
                    elif t.get("turn_type") == "user" and current:
                        groups.append(({"round": rid}, current))
                        current, rid = [], rid + 1
                if current:
                    groups.append(({"round": rid}, current))
            else:
                # all-pairs / adjacent-agent don't need round structure.
                responses = [(t["agent_name"], _clean_response(t["response"]))
                             for t in turns]
                groups = [({}, responses)]
        elif fmt == "segments":
            for seg_idx, segment in enumerate(conversation_data.get("segments", [])):
                responses = []
                for r in segment.get("responses", []):
                    text = _clean_response(r.get("response", ""))
                    if text:
                        responses.append((r["agent_key"], text))
                groups.append(({"segment": seg_idx}, responses))
        else:  # agent_conversations
            agent_convs = conversation_data.get("agent_conversations", {})
            agent_names = list(agent_convs.keys())
            if not agent_names:
                return []
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
                groups.append(({"round": round_idx}, round_responses))
        return groups

    def _extract_pairs(
        self, conversation_data: Dict, agents: Dict, dataset_name: str
    ) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        """Extract agent-response pairs per the configured pairing mode.

        Pairs are emitted in temporal order (earlier response first), which is
        what the 'temporal' NLI direction relies on.
        """
        groups = self._extract_groups(conversation_data)
        pairs: List[Tuple[str, str]] = []
        meta: List[Dict] = []

        if self.pairing_mode == "same-round":
            for group_meta, responses in groups:
                for (a1, r1), (a2, r2) in combinations(responses, 2):
                    pairs.append((r1, r2))
                    meta.append({**group_meta, "agent_a": a1, "agent_b": a2})
        else:
            flat = [item for _, responses in groups for item in responses]
            if self.pairing_mode == "adjacent-agent":
                for (a1, r1), (a2, r2) in zip(flat, flat[1:]):
                    pairs.append((r1, r2))
                    meta.append({"agent_a": a1, "agent_b": a2})
            else:  # all-pairs
                for (a1, r1), (a2, r2) in combinations(flat, 2):
                    pairs.append((r1, r2))
                    meta.append({"agent_a": a1, "agent_b": a2})

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


def main():
    parser = argparse.ArgumentParser(
        description="Automatic (non-LLM) metrics: NLI contradiction + Self-BLEU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=str, required=True,
                        help="MASCOT episode JSONL (one conversation per line)")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSON path for the metric report")
    parser.add_argument("--nli-model", type=str, default=DEFAULT_NLI_MODEL,
                        help="HF model id for NLI classification")
    parser.add_argument("--pairing-mode", type=str, default="same-round",
                        choices=list(PAIRING_MODES),
                        help="How agent responses are paired for NLI "
                             "(same-round is the paper-style default and "
                             "requires round_id metadata)")
    parser.add_argument("--nli-direction", type=str, default="temporal",
                        choices=list(NLI_DIRECTIONS),
                        help="temporal = earlier turn is premise (paper "
                             "definition); bidirectional = optional robustness "
                             "analysis aggregating both directions")
    parser.add_argument("--allow-round-inference", action="store_true",
                        help="Permit inferring rounds from user-turn boundaries "
                             "for legacy files without round_id (loudly logged; "
                             "never applied silently)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-nli", action="store_true")
    parser.add_argument("--skip-self-bleu", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        raise SystemExit(2)

    conversations = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))
    if args.max_samples is not None:
        conversations = conversations[:args.max_samples]
    if not conversations:
        logger.error(f"No conversations found in {input_file}")
        raise SystemExit(2)

    input_metadata = conversations[0].get("metadata") or {}
    logger.info(f"Loaded {len(conversations)} conversations from {input_file}")

    report: Dict[str, Any] = {
        "input_file": str(input_file),
        "num_conversations": len(conversations),
        "input_metadata": input_metadata,
    }

    if not args.skip_nli:
        evaluator = NLIContradictionEvaluator(
            device=args.device, batch_size=args.batch_size,
            model_name=args.nli_model, direction=args.nli_direction,
            pairing_mode=args.pairing_mode,
            allow_round_inference=args.allow_round_inference,
        )
        try:
            nli_results = evaluator.evaluate_batch(conversations, agents={},
                                                   dataset_name="")
        except MissingRoundIdError as e:
            logger.error(str(e))
            raise SystemExit(2)

        total_pairs = sum(r["total_pairs"] for r in nli_results)
        label_counts = {"contradiction": 0, "neutral": 0, "entailment": 0}
        for r in nli_results:
            for d in r["pair_details"]:
                label_counts[d["nli_label"]] += 1
        report["nli"] = {
            "nli_model": args.nli_model,
            "pairing_mode": args.pairing_mode,
            "direction": (args.nli_direction if args.nli_direction == "temporal"
                          else "bidirectional-aggregated"),
            "total_pairs": total_pairs,
            "label_counts": label_counts,
            "mean_contradiction_rate": (
                sum(r["contradiction_rate"] for r in nli_results) / len(nli_results)),
            "mean_entailment_rate": (
                sum(r["entailment_rate"] for r in nli_results) / len(nli_results)),
            "mean_neutral_rate": (
                sum(r["neutral_rate"] for r in nli_results) / len(nli_results)),
            "per_conversation": nli_results,
        }
        logger.info(f"NLI: {total_pairs} pairs, label counts {label_counts}")

    if not args.skip_self_bleu:
        bleu_eval = SelfBLEUEvaluator()
        bleu_results = [bleu_eval.evaluate_conversation(conv, agents={},
                                                        dataset_name="")
                        for conv in conversations]
        scored = [r for r in bleu_results if r["mean_self_bleu"] > 0 or r["per_agent_self_bleu"]]
        report["self_bleu"] = {
            "mean_self_bleu": (sum(r["mean_self_bleu"] for r in bleu_results)
                               / len(bleu_results)),
            "num_conversations_scored": len(scored),
            "per_conversation": bleu_results,
        }
        logger.info(f"Self-BLEU: mean {report['self_bleu']['mean_self_bleu']:.4f}")

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Wrote metric report to {output_file}")


if __name__ == "__main__":
    main()
