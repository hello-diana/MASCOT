"""Judge invocation for the data-generation pipelines.

Covers the three annotation shapes (individual per-query scoring, group
episode+turn scoring, branching batched episode scoring) and the optional LLM
preference-pair verification. Rubric wording, retry counts, and judge
temperatures are owned by src/eval/rubric_evaluator.py and unchanged.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from tqdm import tqdm

from src.data_generation.common import clean
from src.models.llm_models import VLLMServerModel
from src.utils.run_metadata import record_failure

logger = logging.getLogger(__name__)


def verify_pairs_with_llm(
    pairs: List[dict],
    llm: VLLMServerModel,
) -> List[dict]:
    """Use LLM to verify preference pairs; discard disagreements.

    For each pair, ask the LLM which response is better.  If the LLM says
    the rejected response is better, drop the pair.
    """
    if not pairs:
        return pairs

    prompts = []
    for p in pairs:
        chosen_text = p["chosen"][-1]["content"]
        rejected_text = p["rejected"][-1]["content"]
        context = p["chosen"][0]["content"]
        prompt = (
            f"Given the following context, which response is better?\n\n"
            f"Context:\n{context[:2048]}\n\n"
            f"Response A:\n{clean(chosen_text)[:1024]}\n\n"
            f"Response B:\n{clean(rejected_text)[:1024]}\n\n"
            f"Answer with just \"A\" or \"B\"."
        )
        prompts.append(prompt)

    logger.info(f"Verifying {len(prompts)} preference pairs with LLM...")
    verdicts = [(t or "").strip() for t in llm.generate_batch(
        prompts=prompts, temperature=0.0, max_tokens=8,
        system_prompt="You are a helpful judge. Answer with just A or B.",
        desc=f"Pair verification ({len(prompts)})",
    )]

    kept = []
    for pair, verdict in zip(pairs, verdicts):
        v = verdict.strip().upper()
        if "A" in v and "B" not in v:
            kept.append(pair)
        # else: LLM disagrees or ambiguous → drop
    agreement_rate = len(kept) / len(pairs) if pairs else 0
    logger.info(f"LLM verification: kept {len(kept)}/{len(pairs)} pairs "
                f"(agreement rate {agreement_rate:.1%})")
    return kept


def annotate_individual_queries(queries, evaluator, agents, dataset_name,
                                args, ann_file) -> None:
    """Resume-aware parallel annotation of individual queries.

    Mutates each query in *queries* in place, adding a ``scores`` list.
    Prior annotations in *ann_file* are reused when (id, persona) and the
    responses still match; the file is rewritten with the valid rows and each
    newly annotated query is appended as it completes.
    """
    def _annotate_query(qi_q):
        qi, q = qi_q
        scores = []
        for resp in q["responses"]:
            record = {
                "conversation_context": q["context"],
                "agent_response": resp or "",
                "agent_name": q["persona"],
                "id": q["id"],
                "start_idx": None,
                "turn_type": "agent",
            }
            try:
                results = evaluator.evaluate_batch([record], agents, dataset_name)
                scores.append(results[0].total_reward if results else None)
            except Exception as e:
                logger.warning(f"Annotation failed for query {qi}: {e}")
                record_failure(getattr(args, "_counters", None), "num_annotation_failures",
                               getattr(args, "strict", False), f"query {qi}: {e}")
                scores.append(None)
        return qi, scores

    os.makedirs(os.path.dirname(ann_file), exist_ok=True)

    expected = {(q["id"], q["persona"]): qi for qi, q in enumerate(queries)}
    done: Dict[Tuple[str, str], List] = {}
    if os.path.exists(ann_file):
        with open(ann_file) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated tail line from a crash
                key = (row.get("id"), row.get("persona"))
                qi = expected.get(key)
                scores = row.get("scores")
                if (qi is not None and scores
                        and any(s is not None for s in scores)
                        and row.get("responses") == queries[qi]["responses"]):
                    done[key] = scores
        logger.info(f"Resuming annotation: {len(done)}/{len(queries)} queries already done in {ann_file}")

    pending = []
    for qi, q in enumerate(queries):
        key = (q["id"], q["persona"])
        if key in done:
            q["scores"] = done[key]
        else:
            pending.append(qi)

    # Rewrite with validated rows only (dedupes and repairs truncation), then append.
    with open(ann_file, "w") as f:
        for q in queries:
            if "scores" in q:
                f.write(json.dumps(q, default=str) + "\n")

    if pending:
        max_workers = min(args.max_concurrent_requests, len(pending))
        logger.info(f"Annotating {len(pending)} queries with {max_workers} concurrent workers...")
        with open(ann_file, "a") as fout, ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_annotate_query, (qi, queries[qi])): qi
                       for qi in pending}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Annotating individual responses"):
                qi, scores = future.result()
                queries[qi]["scores"] = scores
                fout.write(json.dumps(queries[qi], default=str) + "\n")
                fout.flush()
    logger.info(f"Saved annotated data to {ann_file}")


def annotate_group_episodes(all_episode_states, group_evaluator,
                            individual_evaluator, agents, dataset_name,
                            args, ann_file) -> None:
    """Parallel episode-level + turn-level annotation for the independent
    group pipeline. Mutates episode dicts in place (episode_score,
    turn_scores) and writes the annotated JSONL.

    --max-samples limits QUERIES, never sibling episodes of one query:
    truncating the flat task list would silently drop the episodes needed
    to form same-query preference pairs.
    """
    query_keys = list(all_episode_states.keys())
    if args.max_samples:
        query_keys = query_keys[:args.max_samples]
    annotation_tasks = []
    for qk in query_keys:
        for ep_idx, ep in enumerate(all_episode_states[qk]):
            annotation_tasks.append((qk, ep_idx, ep))

    logger.info(f"Annotating {len(annotation_tasks)} episodes")

    def _annotate_episode(task):
        qk, ep_idx, ep = task
        # Episode-level score
        try:
            results = group_evaluator.evaluate_batch(ep, agents)
            episode_score = results[0].total_reward if results else None
        except Exception as e:
            logger.warning(f"Group eval failed for {qk}: {e}")
            record_failure(getattr(args, "_counters", None), "num_annotation_failures",
                           getattr(args, "strict", False), f"group eval {qk}: {e}")
            episode_score = None

        # Turn-level scores
        eval_data = []
        for turn in ep.get("turns", []):
            aname = turn.get("agent_name", "unknown")
            if aname not in agents:
                continue
            eval_data.append({
                "conversation_context": ep.get("initial_situation", "") or
                                        ep.get("meeting", {}).get("meeting_context", ""),
                "agent_response": turn.get("response", ""),
                "agent_name": aname,
                "id": qk,
                "start_idx": ep.get("start_idx"),
                "turn_type": turn.get("turn_type", "agent"),
            })

        if eval_data:
            try:
                turn_results = individual_evaluator.evaluate_batch(eval_data, agents, dataset_name)
                turn_scores = [r.total_reward for r in turn_results]
            except Exception as e:
                logger.warning(f"Turn eval failed for {qk}: {e}")
                record_failure(getattr(args, "_counters", None), "num_annotation_failures",
                               getattr(args, "strict", False), f"turn eval {qk}: {e}")
                turn_scores = [None] * len(eval_data)
        else:
            turn_scores = []

        return qk, ep_idx, episode_score, turn_scores

    os.makedirs(os.path.dirname(ann_file), exist_ok=True)

    max_workers = min(args.max_concurrent_requests, len(annotation_tasks))
    logger.info(f"Annotating {len(annotation_tasks)} episodes with {max_workers} concurrent workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_annotate_episode, t): t for t in annotation_tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Annotating episodes"):
            qk, ep_idx, episode_score, turn_scores = future.result()
            all_episode_states[qk][ep_idx]["episode_score"] = episode_score
            all_episode_states[qk][ep_idx]["turn_scores"] = turn_scores

    with open(ann_file, "w") as f:
        for qk, episodes in all_episode_states.items():
            f.write(json.dumps({"query_key": qk, "episodes": episodes}, default=str) + "\n")
    logger.info(f"Saved annotated episodes to {ann_file}")


def annotate_branching_states(completed_states, group_evaluator, judge_llm,
                              agents, args) -> None:
    """Single batched episode-level evaluation for branching states.

    Mutates each state in place, setting ``episode_score`` (None on parse
    failure or empty prompt).
    """
    eval_prompts = []
    eval_meta = []  # (key, orig, clean)
    for key, state in completed_states.items():
        prompt, orig, cleaned = group_evaluator.build_prompt(state, agents)
        if prompt is not None:
            eval_prompts.append(prompt)
            eval_meta.append((key, orig, cleaned))
        else:
            state["episode_score"] = None

    if eval_prompts:
        eval_results = judge_llm.generate_batch(
            prompts=eval_prompts, temperature=0.0, max_tokens=3000,
            desc=f"Group eval ({len(eval_prompts)} episodes)",
        )
        for i, (key, orig, cleaned) in enumerate(eval_meta):
            state = completed_states[key]
            try:
                parsed = group_evaluator.parse_result(
                    eval_results[i], state, orig, cleaned)
                state["episode_score"] = (
                    parsed[0].total_reward if parsed else None)
            except Exception as e:
                logger.warning(f"Eval parse failed for {key}: {e}")
                record_failure(getattr(args, "_counters", None), "num_annotation_failures",
                               getattr(args, "strict", False), f"eval parse {key}: {e}")
                state["episode_score"] = None
