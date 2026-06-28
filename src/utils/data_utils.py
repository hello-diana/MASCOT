
import os
import json
from typing import Dict, List, Optional
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer
import logging
import re
from datetime import datetime
from pathlib import Path
from config.persona_config import AGENTS
from src.const import (
    EMPATHETIC_DATASET,
    ESCONV_DATASET,
    QMSUM_DATASET,
    DATASET2HF,
)
from src.prompts.persona_templates import MascotPersonaTemplates
logger = logging.getLogger(__name__)

def read_jsonl(f):
    return [json.loads(x) for x in open(f).readlines()]

def read_json(f):
    return json.load(open(f, 'r'))

def clean_response(response: str) -> str:
    """Remove <think> tags from responses for readability."""
    if not response:
        return ''
    # Remove <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    return cleaned.strip()

def get_file_timestamps(file_path):
    """Get file creation and modification timestamps.

    Args:
        file_path: Path to the file (str or Path object)

    Returns:
        dict with 'created' and 'modified' datetime objects and formatted strings
    """
    file_path = Path(file_path)
    stat = file_path.stat()

    # Get creation time (ctime on Unix is actually the last metadata change time,
    # but it's the closest we have to creation time on Linux)
    created_timestamp = stat.st_ctime
    modified_timestamp = stat.st_mtime
    created_dt = datetime.fromtimestamp(created_timestamp)
    modified_dt = datetime.fromtimestamp(modified_timestamp)
    
    created_str = created_dt.strftime('%Y-%m-%d %H:%M:%S')
    modified_str = modified_dt.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"  Created: {created_str}")
    print(f"  Last modified: {modified_str}")

    return {
        'created': created_dt,
        'modified': modified_dt,
        'created_str': created_str,
        'modified_str': modified_str
    }

def write_jsonl(data: list, f):
    with open(f, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def load_meeting_prompts_dataset(data_path: str, tokenizer, dataset_name: str = QMSUM_DATASET,
                                 turns_per_round: int = 100,
                                 score_margin: float = 0.5) -> Dataset:
    """Load meeting transcripts and create prompts for RL training.

    Returns dataset with:
    - prompt: meeting transcript + persona role instruction
    - persona: persona key (e.g., "minutes_scribe")
    - meeting_id: QMSum meeting ID

    Scores are kept in native [1, 5] rubric range (no rescaling).
    Uses all-pairs strategy: for every (i, j) with i < j in score-sorted
    candidates, create a pair if score_diff >= score_margin.
    """
    prompts = []

    with open(data_path, 'r') as f:
        for line in f:
            meeting_data = json.loads(line)
            transcript = meeting_data["original_transcript"]
            meeting_id = meeting_data["meeting_id"]
            meeting_summary = meeting_data["meeting_summary"]
            turns_per_round = meeting_data.get("turns_per_round", 100)

            # Process each persona and their segments
            for persona_key in AGENTS.keys():
                if persona_key not in meeting_data["scored_agents"]:
                    continue

                segments = meeting_data["scored_agents"][persona_key]

                for segment_id in range(len(segments)):
                    # Format transcript for this segment
                    start_turn = segment_id * turns_per_round
                    end_turn = min(start_turn + turns_per_round, len(transcript))
                    formatted_transcript = ""
                    for turn in transcript[start_turn:end_turn]:
                        formatted_transcript += f"{turn['speaker']}: {turn['text']}\n"

                    # Create prompt
                    persona_spec = AGENTS[persona_key]
                    prompt_text = MascotPersonaTemplates.speaker_prompt(
                        dataset_name, persona_spec, formatted_transcript, meeting_summary
                    )

                    candidates = meeting_data["scored_agents"][persona_key][segment_id]['candidates']
                    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)

                    # All-pairs: for every (i, j) with i < j, create pair if margin met
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            score_chosen = candidates[i]['score']
                            score_rejected = candidates[j]['score']
                            if score_chosen - score_rejected < score_margin:
                                continue

                            chosen_response = candidates[i]['response']
                            rejected_response = candidates[j]['response']
                            if not chosen_response or not rejected_response:
                                continue

                            chosen = [
                                {"role": "user", "content": prompt_text},
                                {"role": "assistant", "content": chosen_response}
                            ]
                            rejected = [
                                {"role": "user", "content": prompt_text},
                                {"role": "assistant", "content": rejected_response}
                            ]

                            prompts.append({
                                "chosen": chosen,
                                "rejected": rejected,
                                "score_chosen": score_chosen,
                                "score_rejected": score_rejected,
                                "persona": persona_key,
                                "meeting_id": meeting_id
                            })

    # Create dataset from prompts (no score rescaling — native [1, 5] range)
    dataset = Dataset.from_list(prompts)

    logger.info(f"Created {len(dataset)} preference pairs (scores in native [1, 5] range)")
    return dataset



def load_group_reward_dataset(run_files: list, score_margin: float = 0.5) -> Dataset:
    """Convert group episode data into preference pairs.

    Args:
        run_files: List of JSONL file paths containing group episode data.
            Each line: {"query_key": str, "episodes": [episode_dict, ...]}
        score_margin: Minimum score difference for a valid pair.

    Returns:
        HF Dataset with columns: chosen, rejected, score_chosen, score_rejected,
        persona, meeting_id.
    """
    # Index episodes by query_key
    episodes_by_key = {}
    for fpath in run_files:
        for line in open(fpath):
            rec = json.loads(line)
            qk = rec["query_key"]
            episodes_by_key.setdefault(qk, []).extend(rec["episodes"])

    pairs = []
    for qk, episodes in episodes_by_key.items():
        if len(episodes) < 2:
            continue

        sample_ep = episodes[0]
        prompt_context = sample_ep.get("meeting", {}).get("meeting_context", "")
        ep_id = sample_ep.get("id", qk)

        # Sort episodes by score descending
        scored = [(i, ep.get("episode_score", 3.0)) for i, ep in enumerate(episodes)]
        scored.sort(key=lambda x: x[1], reverse=True)

        for a_pos in range(len(scored)):
            for b_pos in range(a_pos + 1, len(scored)):
                idx_a, score_a = scored[a_pos]
                idx_b, score_b = scored[b_pos]
                if score_a - score_b < score_margin:
                    continue

                # Full episode text — all turns concatenated (with <think> tags preserved)
                chosen_text = "\n".join(
                    f"[{t.get('name', t.get('agent_name', 'unknown'))}]: {t.get('response', '')}"
                    for t in episodes[idx_a].get("turns", [])
                )
                rejected_text = "\n".join(
                    f"[{t.get('name', t.get('agent_name', 'unknown'))}]: {t.get('response', '')}"
                    for t in episodes[idx_b].get("turns", [])
                )
                if not chosen_text.strip() or not rejected_text.strip():
                    continue

                pairs.append({
                    "chosen": [
                        {"role": "user", "content": prompt_context},
                        {"role": "assistant", "content": chosen_text},
                    ],
                    "rejected": [
                        {"role": "user", "content": prompt_context},
                        {"role": "assistant", "content": rejected_text},
                    ],
                    "score_chosen": score_a,
                    "score_rejected": score_b,
                    "persona": "group",
                    "meeting_id": ep_id,
                })

    dataset = Dataset.from_list(pairs) if pairs else Dataset.from_dict({
        "chosen": [], "rejected": [], "score_chosen": [], "score_rejected": [],
        "persona": [], "meeting_id": [],
    })
    logger.info(f"Created {len(dataset)} group preference pairs from {len(episodes_by_key)} queries")
    return dataset


def create_or_load_rl_dataset(data_path, tokenizer, source_jsonl=None, dataset_name=QMSUM_DATASET):

    if os.path.exists(data_path):
        dataset = load_from_disk(data_path)
    else:
        jsonl_path = source_jsonl or "data/reward_model/rm_conversations_QMSum_Qwen3-8B.jsonl"
        dataset = load_meeting_prompts_dataset(jsonl_path, tokenizer, dataset_name=dataset_name, turns_per_round=100)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        dataset.save_to_disk(data_path)

    return dataset


# ============================================================================
# Per-utterance dataset loaders
# ============================================================================

def load_empathetic_dialogues(
    split: str = 'train',
    dataset_name: str = DATASET2HF[EMPATHETIC_DATASET],
    local_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """Load EmpatheticDialogues from Hugging Face or local CSV files.

    Returns one dict per *first* utterance of each conversation. Fields:
        conv_id, utterance_idx, context, prompt, speaker_idx,
        utterance, selfeval, tags
    """
    original_split = split
    if split == 'validation':
        split = 'valid'

    print(f'Loading EmpatheticDialogues dataset (split: {original_split})...')

    if local_path:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local dataset path not found: {local_path}")
        print(f'  Loading from local path: {local_path}')
        csv_path = local_path
        use_csv = True
    else:
        try:
            print(f'  Loading from Hugging Face: {dataset_name}')
            data = load_dataset(dataset_name, split=original_split, cache_dir=cache_dir)
            use_csv = False
        except Exception as e:
            print(f'  Note: Could not load from HF ({str(e)[:50]}...)')
            print(f'  Downloading from original Facebook source instead...')
            csv_path = "data/empathetic_dialogues"
            use_csv = True

    if use_csv:
        try:
            csv_file = os.path.join(csv_path, f'{split}.csv')
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"CSV file not found: {csv_file}")
            print(f'  Loading CSV file: {csv_file}')
            df = pd.read_csv(csv_file, quoting=3, on_bad_lines='skip', engine='python')
            data = Dataset.from_pandas(df)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

    utterances = []
    conv_ids = set()
    for item in data:
        utterance_dict = dict(item)
        if utterance_dict["conv_id"] in conv_ids:
            continue
        conv_ids.add(utterance_dict["conv_id"])
        utterances.append(utterance_dict)

    print(f'Loaded {len(utterances):,} utterances from {original_split} split')
    return utterances


def load_esconv_dialogues(
    split: str = "train",
    dataset_name: str = DATASET2HF[ESCONV_DATASET],
    cache_dir: Optional[str] = None,
) -> List[Dict]:
    """Load ESConv and flatten to the same per-utterance shape as load_empathetic_dialogues.

    Speaker mapping:
        'usr' -> speaker_idx '1' (help-seeker)
        'sys' -> speaker_idx '0' (supporter)
    """
    hf_split = "validation" if split in ("valid", "validation") else split

    print(f"Loading ESConv dataset (split: {hf_split}) from {dataset_name}...")
    data = load_dataset(dataset_name, split=hf_split, cache_dir=cache_dir)

    utterances: List[Dict] = []
    for row_idx, row in enumerate(data):
        conv_id = f"esconv_{hf_split}_{row_idx}"
        situation = row.get("situation") or ""
        emotion = row.get("emotion_type") or ""
        problem = row.get("problem_type") or ""
        survey = row.get("survey_score") or {}
        try:
            selfeval_str = json.dumps(survey, ensure_ascii=False)
        except Exception:
            selfeval_str = str(survey)

        dialog = row.get("dialog") or []
        for turn_idx, turn in enumerate(dialog):
            speaker = (turn.get("speaker") or "").lower()
            text = turn.get("text") or ""
            speaker_idx = "1" if speaker == "usr" else "0"
            utterances.append({
                "conv_id": conv_id,
                "utterance_idx": str(turn_idx + 1),
                "context": emotion,
                "prompt": situation,
                "speaker_idx": speaker_idx,
                "utterance": text,
                "selfeval": selfeval_str,
                "tags": problem,
            })

    print(f"Loaded {len(utterances):,} utterances from {hf_split} split "
          f"({len(data):,} conversations)")
    return utterances


# ============================================================================
# Per-conversation "items" loaders for the unified eval pipeline
# ============================================================================

def load_qmsum_items(split: str, max_samples: Optional[int]) -> List[Dict]:
    meetings = load_dataset(DATASET2HF[QMSUM_DATASET], split=split)
    if max_samples:
        meetings = meetings.select(range(min(max_samples, len(meetings))))
    return list(meetings)


def extract_unique_conversations(utterances: List[Dict]) -> List[Dict]:
    conversations: Dict[str, Dict] = {}
    for u in utterances:
        cid = u['conv_id']
        if cid not in conversations:
            conversations[cid] = {
                'conv_id': cid,
                'context': u['context'],
                'prompt': u['prompt'],
            }
    return list(conversations.values())


def load_empathetic_situations(
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """One dict per conversation: {conv_id, situation, emotion}.

    Reads DianaW/empathetic_dialogues and dedups by conv_id during a single
    pass — avoids the load-all-utterances-then-dedup roundtrip.
    """
    ds = load_dataset(DATASET2HF[EMPATHETIC_DATASET], split=split)
    seen = set()
    contexts: List[Dict] = []
    for row in ds:
        cid = row["conv_id"]
        if cid in seen:
            continue
        seen.add(cid)
        contexts.append({
            "conv_id": cid,
            "situation": row["prompt"],
            "emotion": row["context"],
        })
        if max_samples and len(contexts) >= max_samples:
            break
    return contexts


def load_esconv_situations(
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """One dict per ESConv conversation: {conv_id, situation, emotion}.

    Reads Ahren09/ESConv per-row directly; no utterance flattening.
    """
    hf_split = "validation" if split in ("valid", "validation") else split
    ds = load_dataset(DATASET2HF[ESCONV_DATASET], split=hf_split)
    contexts: List[Dict] = []
    for i, row in enumerate(ds):
        contexts.append({
            "conv_id": f"esconv_{hf_split}_{i}",
            "situation": row.get("situation") or "",
            "emotion": row.get("emotion_type") or "",
        })
        if max_samples and len(contexts) >= max_samples:
            break
    return contexts


def load_empathetic_items(split: str, dataset_name: str, local_path: Optional[str],
                          max_samples: Optional[int]) -> List[Dict]:
    utterances = load_empathetic_dialogues(
        split=split,
        dataset_name=dataset_name,
        local_path=local_path,
    )
    conversations = extract_unique_conversations(utterances)
    if max_samples:
        conversations = conversations[:min(max_samples, len(conversations))]
    return conversations


def load_esconv_items(split: str, max_samples: Optional[int]) -> List[Dict]:
    """Load ESConv conversations via the empathetic-schema adapter."""
    utterances = load_esconv_dialogues(split=split)
    conversations = extract_unique_conversations(utterances)
    if max_samples:
        conversations = conversations[:min(max_samples, len(conversations))]
    return conversations


# ============================================================================
# MultiAgentESC baseline loaders (different shape: pre-formatted contexts)
# ============================================================================

def load_empathetic(max_samples: Optional[int] = None) -> List[Dict]:
    """Load EmpatheticDialogues test set, one initial-turn dict per conversation.

    Returns: list of {conv_id, context, initial_situation}.
    """
    ds = load_dataset(DATASET2HF[EMPATHETIC_DATASET], split="test")

    samples: List[Dict] = []
    conv_ids = set()
    for row in ds:
        if row['conv_id'] in conv_ids:
            continue
        conv_ids.add(row['conv_id'])
        context = f"User: {row['prompt']}".replace("_comma_", ",").strip()

        samples.append({
            "conv_id": row['conv_id'],
            "context": context,
            "initial_situation": context,
        })

        if max_samples is not None and len(samples) >= max_samples:
            break
    logger.info("Loaded %d EmpatheticDialogues conversations", len(samples))
    return samples


def load_esconv(max_samples: Optional[int] = None) -> List[Dict]:
    """Load ESConv test set, one initial-turn dict per conversation.

    Returns: list of {conv_id, context, initial_situation, initial_emotion}.
    Schema matches load_empathetic() so MultiAgentESC's pipeline consumes it
    without any further changes.
    """
    ds = load_dataset(DATASET2HF[ESCONV_DATASET], split="test")

    samples: List[Dict] = []
    for row_idx, row in enumerate(ds):
        situation = (row.get("situation") or "").strip()
        emotion = (row.get("emotion_type") or "").strip()
        if not situation:
            continue
        context = f"User: {situation}"
        samples.append({
            "conv_id": f"esconv_test_{row_idx}",
            "context": context,
            "initial_situation": context,
            "initial_emotion": emotion,
        })
        if max_samples is not None and len(samples) >= max_samples:
            break
    logger.info("Loaded %d ESConv conversations", len(samples))
    return samples


def load_qmsum(max_samples: Optional[int] = None) -> List[Dict]:
    """Load QMSum test set as one sample per query over a meeting transcript.

    Returns: list of {conv_id, context, reference, query, id, start_idx, meeting}.
    """
    ds = load_dataset(DATASET2HF[QMSUM_DATASET], split="test")
    samples: List[Dict] = []
    for row in ds:
        transcript_turns = row["meeting_transcripts"]
        turns = [f"{t['speaker']}: {t['content'].strip()}" for t in transcript_turns]
        transcript_text = " ".join(turns)

        for q in row.get("general_query_list", []) + row.get("specific_query_list", []):
            query = q["query"]
            answer = q["answer"]
            context = transcript_text + f" User: {query}"
            samples.append({
                "conv_id": row["id"],
                "context": context,
                "reference": answer,
                "query": query,
                "id": row["id"],
                "start_idx": 0,
                "meeting": {"meeting_context": transcript_text},
            })

    if max_samples:
        samples = samples[:max_samples]
    logger.info("Loaded %d QMSum query-answer pairs", len(samples))
    return samples
        
        
    

# ---------------------------------------------------------------------------
# Shared output-path construction (single source of truth; the byte-exact
# formats are load-bearing — downstream analysis and resume logic match on them)
# ---------------------------------------------------------------------------

def mascot_output_path(args, dataset: str) -> Path:
    """MASCOT eval .jsonl path for both QMSum and Empathetic/ESConv runs.

    Reproduces the historical eval_mascot_qmsum / eval_mascot_empathetic
    filename formats byte-identically (incl. the optional --suffix, absent on
    launcher Namespaces).
    """
    from src.const import QMSUM_DATASET, EMPATHETIC_DATASET, ESCONV_DATASET

    directive = Path(args.directive_model).name.replace('/', '_')
    agent = Path(args.agent_model).name.replace('/', '_')
    user = Path(args.user_model).name.replace('/', '_') if args.enable_user_simulation else None
    extra = getattr(args, 'suffix', None)

    if dataset == QMSUM_DATASET:
        out_dir = Path(args.output_dir) / "mascot_eval" / "QMSum" / args.split
        suffix = f"_{extra}" if extra else ""
        if args.no_thinking:
            suffix += "_no-thinking"
        if args.mode != "mascot":
            suffix += f"_{args.mode}"
        if args.max_samples:
            suffix += f"_max{args.max_samples}"
        filename = f"{args.mode}_default-persona_{directive}_{agent}_{user}{suffix}.jsonl"
        return out_dir / filename

    # Empathetic / ESConv
    ds_label = ESCONV_DATASET if dataset == ESCONV_DATASET else EMPATHETIC_DATASET
    out_dir = Path(args.output_dir) / "mascot_eval" / ds_label / args.split

    if args.enable_big_five:
        mode_str = "big5-persona"
    else:
        mode_str = "default-persona"

    suffix = f"_{extra}" if extra else ""
    suffix += "_no_user" if not args.enable_user_simulation else ""
    suffix += f"_max{args.max_samples}" if args.max_samples else ""
    filename = f"{args.mode}_{mode_str}_{directive}_{agent}_{user}{suffix}.jsonl"
    return out_dir / filename


def judge_metrics_path(input_file, judge: str, output_dir=None, suffix: str = "") -> Path:
    """Rubric-judge metrics .json path (used by llm_as_a_judge)."""
    input_file = Path(input_file)
    base_name = input_file.name.split('.json')[0]
    out_dir = Path(output_dir) if output_dir else input_file.parent
    return out_dir / f"metrics_{base_name}_judge-{judge.split('/')[-1]}{suffix}.json"
