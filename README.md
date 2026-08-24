## 📖 Overview

Social interaction shapes well-being, mental health, and cognition — yet an estimated **1 in 6** people worldwide lack the social contact they need. As large language models move from passive **tools** to active **socio-collaborative companions**, **MASCOT** is a generalizable multi-agent framework for building *multi-perspective* companions that balance **individual persona consistency** with **global discourse dynamics**.

MASCOT uses an efficient **bi-level optimization** strategy that targets two recurring failures of multi-agent LLM setups:

- 🎭 **Persona Fidelity** — a Reinforcement-Learning-from-AI-Feedback (RLAIF) pipeline fine-tunes each agent for a strict, unique persona (via **GRPO**), preventing *persona collapse* — agents losing their distinct roles and regressing to generic assistant behavior.
- 🤝 **Interaction Synergy** — a director / meta-agent policy guided by **group-level rewards** coordinates turn-taking for complementary, non-redundant dialogue, avoiding *social sycophancy* and echo chambers — agents repeating each other or agreeing unproductively instead of contributing complementary content.

> 💡 Unlike prior multi-agent systems optimized purely for task efficiency, MASCOT targets **user–agent interaction quality** in affective and collaborative settings (emotional support and workplace meetings).

## 📊 Results

| Metric                                      |      Result      |
| :------------------------------------------ | :--------------: |
| 🎭 Persona Consistency                      | **+14.1** |
| 🤝 Social Contribution                      | **+10.6** |
| 👥 Human preference (head-to-head)          |  **69%**  |
| ⚡ Trainable policy parameters (LoRA, r=16) | **0.187%** |

- 🔬 **Extensive evaluation** — human, multi-judge (GPT-4o / Gemma-3-27B / Phi-4), and automatic metrics across **Empathetic Dialogues** and **QMSum** (workplace meetings), plus out-of-domain **ESConv**.

## ✨ Highlights

- 🧩 **Two-stage framework** — persona-aware behavioral alignment of individual Speakers, followed by collaborative dialogue optimization of the full group.
- 🪶 **Low-resource bi-level optimization** — Speaker and Director **policies** train with LoRA adapters only (base weights are referenced by Hugging Face name). Reward-model checkpoints and explicitly merged checkpoints may contain full model weights — see [Training](#-training).
- 🎬 **Director orchestration** — a Director agent selects who speaks next and issues natural-language directives specifying *speaker, action, and tone*. The Director is optimized with **GRPO** (the paper's method); an offline **DPO** trainer is retained as an optional alternative.
- 📦 **Self-contained pipeline** — episode generation, rubric judging, preference-data construction, reward modeling, GRPO/DPO training, and evaluation, all against an OpenAI-compatible vLLM server.

## 🧩 The MASCOT Framework

MASCOT has two stages, each backed by a dedicated entry point under `src/`:

| Pipeline step                                | Stage | Entry point                                                |
| :------------------------------------------- | :---- | :--------------------------------------------------------- |
| 🎨**Per-persona candidate generation** | 1     | `src/dataset/generate_and_annotate.py --mode individual` |
| 🧑‍⚖️**Rubric-based LLM judging**   | 1 + 2 | `src/eval/rubric_evaluator.py`                           |
| 🏅**Persona reward model**             | 1     | `src/train/train_reward_model.py`                        |
| 🎯**Speaker GRPO**                     | 1     | `src/train/train_rl.py`                                  |
| 🗣️**Directed episode generation**    | 2     | `src/generation/mascot_episode.py`                       |
| 🌿**Branching group episodes**         | 2     | `src/dataset/generate_and_annotate.py --mode group`      |
| 🏅**Group reward model**               | 2     | `src/train/train_group.py --mode train-reward-model`     |
| 🎬**Director GRPO** (paper default)    | 2     | `src/train/train_group.py --mode train-director-grpo`    |
| 🎬**Director DPO** (optional)          | 2     | `src/train/train_group.py --mode train-director-dpo`     |

### Stage 1 — Persona-Aware Behavioral Alignment

Speaker agents are aligned to preserve distinct personas:

1. **K = 8** candidate responses are generated per context/persona (`generate_and_annotate.py --mode individual --num-candidates 8`).
2. Candidates are scored with rubric-based feedback by an LLM judge (`src/eval/rubric_evaluator.py`).
3. Preference pairs are built from the scored candidates (up to K·(K−1)/2 = 28 unordered pairs per query before margin filtering), and a scalar **persona reward model** (Qwen3-0.6B) is trained on them.
4. The Speaker policy (Qwen3-8B) is optimized with **GRPO** against the reward model, plus small format and conciseness rewards.

### Stage 2 — Collaborative Dialogue Optimization

A **Director** agent orchestrates the conversation:

1. The Director selects which Speaker should talk next and emits a natural-language directive specifying *speaker, action, and tone*.
2. The selected Speaker responds conditioned on its persona, the directive, and the conversation history (`src/generation/mascot_episode.py`).
3. Full trajectories are scored by a **group reward model** (Qwen3-0.6B), and the Director (Qwen3-8B) is optimized with **GRPO** over trajectory rewards (`train_group.py --mode train-director-grpo`) — this is the paper's primary Stage-2 method.

> 🔀 **Director GRPO vs. Director DPO.** `train-director-grpo` performs **online, group-relative optimization** over trajectory rewards and is the method described in the paper (and the default). `train-director-dpo` performs **offline preference optimization** over stored directive/trajectory pairs from branching episodes; it is retained as an optional alternative and is *not* the paper-reproduction path.

## 🎭 Agent Rosters

| Domain                        | Datasets                      | Roster                                                                                                               |
| :---------------------------- | :---------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| 💛**Emotional support** | Empathetic Dialogues / ESConv | *The Anchor* (emotional validator) · *The Catalyst* (action-oriented guide) · *The Beacon* (growth advocate) |
| 📋**Meetings**          | QMSum                         | *Minutes Scribe* · *Decision Logger* · *Action Item Captain* · *Critic*                                   |

Personas and prompts live in `config/persona_config.py` and `src/prompts/persona_templates.py`. Simulated users can optionally carry Big Five (OCEAN) personality profiles (`src/user/simulated_users.py`).

## 🚀 Installation

```bash
pip install -e .                 # core: episode generation + rubric evaluation
pip install -e ".[train]"        # + reward model / GRPO / director training
pip install -e ".[serve]"        # + local vLLM serving
pip install -e ".[metrics]"      # + automatic non-LLM metrics (NLI, Self-BLEU)
pip install -e ".[wandb]"        # + optional Weights & Biases tracking
pip install -e ".[api-judges]"   # + optional Anthropic/Gemini judges
```

| Extra          | Adds                                                                   |
| :------------- | :--------------------------------------------------------------------- |
| *(core)*     | Episode generation + rubric evaluation against a vLLM server           |
| `train`      | Reward-model / GRPO / director training (torch, trl, peft, accelerate) |
| `serve`      | Local vLLM serving (`scripts/serve_vllm.py`)                         |
| `metrics`    | Automatic non-LLM metrics — NLI contradiction, Self-BLEU              |
| `wandb`      | Optional Weights & Biases tracking                                     |
| `api-judges` | Optional Anthropic / Gemini judges                                     |

> 💡 **Weights & Biases is fully optional**: training scripts only initialize it when the `WANDB_PROJECT` environment variable is set (`WANDB_ENTITY` is also read from the environment).

> 💡 **API keys** for optional judges are read from the environment / `.env` (`OPENAI_API_KEY`, `GOOGLE_API_KEY`); nothing is hardcoded.

All commands below run from the repository root with `python -m ...`. Datasets are pulled from Hugging Face automatically (see `DATASET2HF` in `src/const.py`). Generated artifacts go to `data/` and `outputs/`. The canonical model/hyperparameter defaults live in `src/const.py`.

## ⚡ Quick Start — the full pipeline, connected

Create the environment with `pip install -e ".[train,serve,metrics]" -c constraints-phase1.txt`.

### 1️⃣ Serve Qwen3-8B with vLLM

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/serve_vllm.py \
    --model Qwen/Qwen3-8B --gpus 2 --port 8008
```

`--gpus` takes a count (`2`) or an explicit list (`0,1`); tensor parallelism is inferred. Add `--gpu-memory-utilization 0.5` on shared GPUs.

> 🔌 **Endpoints.** Server URLs are never defaulted — pass them explicitly. `--vllm-server-url` is a shared convenience endpoint used for the Director, Speaker, and User; role-specific URLs (`--director-server-url`, `--speaker-server-url`, `--user-server-url`) override it per role. The judge endpoint (`--judge-server-url`) is independent and never inherits the shared URL. `--share-model` (one loaded model reused across roles) is opt-in, never forced.

### 2️⃣ Generate K=8 candidate responses + preference pairs (Stage 1)

```bash
python -m src.dataset.generate_and_annotate \
    --mode individual --dataset QMSum \
    --model-name Qwen/Qwen3-8B \
    --num-candidates 8 --score-margin 0.5 \
    --vllm-server-url http://localhost:8008/v1 \
    --judge-server-url http://localhost:8008/v1 \
    --output-dir data/rl
# -> data/rl/individual_QMSum_Qwen3-8B
```

> Pass `--resume` to continue a run or `--overwrite` to regenerate.

### 3️⃣ Train the Qwen3-0.6B persona reward model

```bash
python -m src.train.train_reward_model \
    --dataset QMSum --model-name Qwen/Qwen3-0.6B \
    --data-path data/rl/individual_QMSum_Qwen3-8B
# -> outputs/reward_model/Individual_QMSum_Qwen3-0.6B  (LoRA adapter + mascot_metadata.json)
```

### 4️⃣ Train the Qwen3-8B Speaker with GRPO

```bash
python -m src.train.train_rl --algorithm grpo --dataset QMSum \
    --data-path data/rl/individual_QMSum_Qwen3-8B \
    --reward-model-path outputs/reward_model/Individual_QMSum_Qwen3-0.6B
# -> outputs/llm/grpo_Speaker_individual_QMSum_Qwen3-8B  (LoRA adapter)
```

### 5️⃣ Generate group trajectories (Stage 2)

```bash
# a) independent episodes -> group reward-model preference pairs
python -m src.dataset.generate_and_annotate \
    --mode group --dataset QMSum \
    --model-name Qwen/Qwen3-8B --generation-mode independent \
    --num-episodes 2 --num-turns 5 \
    --user-response-probability 0.5 --score-margin 0.5 --seed 42 \
    --vllm-server-url http://localhost:8008/v1 \
    --judge-server-url http://localhost:8008/v1 \
    --output-dir data/rl
# -> data/rl/group_QMSum_Qwen3-8B

# b) branching episodes -> Director training data
python -m src.dataset.generate_and_annotate \
    --mode group --dataset QMSum \
    --model-name Qwen/Qwen3-8B --generation-mode branching \
    --num-candidates 8 --num-turns 5 \
    --user-response-probability 0.5 --score-margin 0.5 --seed 42 \
    --vllm-server-url http://localhost:8008/v1 \
    --judge-server-url http://localhost:8008/v1 \
    --output-dir data/rl
# -> data/rl/branching_QMSum_Qwen3-8B
```

### 6️⃣ Train the Qwen3-0.6B group reward model

```bash
python -m src.train.train_group --mode train-reward-model \
    --dataset QMSum --group-reward-model-name Qwen/Qwen3-0.6B \
    --data-path data/rl/group_QMSum_Qwen3-8B
# -> outputs/reward_model/group_reward_QMSum_Qwen3-0.6B  (full-weight checkpoint — see Training note)
```


### 7️⃣ Train the Qwen3-8B Director with GRPO (paper default)

```bash
python -m src.train.train_group --mode train-director-grpo \
    --dataset QMSum \
    --episode-dataset-path data/rl/branching_QMSum_Qwen3-8B \
    --rollout-server-url http://localhost:8008/v1 \
    --use-vllm
```

Optional offline alternative (not the paper path):

```bash
python -m src.train.train_group --mode train-director-dpo \
    --dataset QMSum \
    --episode-dataset-path data/rl/branching_QMSum_Qwen3-8B
# -> outputs/llm/dpo_Director_QMSum_Qwen3-8B  (LoRA adapter)
```

### 8️⃣ Generate final MASCOT conversations with the TRAINED system

Serve the base model plus **both trained adapters** as LoRA aliases on one
multi-LoRA server, then run the three-role system — Director
(`mascot-director`), Speakers (`mascot-speaker`), and the simulated User on
the base `Qwen/Qwen3-8B`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/serve_vllm.py \
    --model Qwen/Qwen3-8B --gpus 2 --port 8009 \
    --lora-modules \
        mascot-director=outputs/llm/grpo_Director_QMSum_Qwen3-8B \
        mascot-speaker=outputs/llm/grpo_Speaker_individual_QMSum_Qwen3-8B

python -m src.eval.eval_mascot_qmsum \
    --dataset QMSum --split test \
    --directive-model mascot-director \
    --agent-model mascot-speaker \
    --user-model Qwen/Qwen3-8B \
    --enable-user-simulation \
    --vllm-server-url http://localhost:8009/v1 \
    --output-dir outputs
# -> outputs/mascot_eval/QMSum/test/*.jsonl
#    (+ per-run sidecar <output>.jsonl.runs/<run_id>/run_config.json / run_summary.json)
```

Each role's request uses its own served model id; episode metadata records the
base model AND the adapter identity per role, and `--trace-role-calls` writes
a per-request `role_call_trace.jsonl` proving which model id every role
actually called. Add `--enable-big-five` (optionally with
`--user-big-five-profile "High Openness" ...`) to simulate users with Big Five
personality profiles on the empathetic datasets.

> 🐛 **Debugging: untrained baseline.** To sanity-check the pipeline without
> trained adapters, point all roles at the base server from step 1:
> `--directive-model Qwen/Qwen3-8B --agent-model Qwen/Qwen3-8B --share-model --vllm-server-url http://localhost:8008/v1`. This generates with the BASE
> model — it is not the trained MASCOT system.
> Alternatively, merge an adapter into full weights with
> `src/train/merge_lora.py` and serve the merged checkpoint directly.

### 9️⃣ Rubric evaluation (LLM-as-a-judge)

```bash
python -m src.eval.llm_as_a_judge \
    --input-file outputs/mascot_eval/QMSum/test/<generated-file>.jsonl \
    --dataset QMSum \
    --provider vllm --judge Qwen/Qwen3-8B \
    --judge-server-url http://localhost:8009/v1
```

### 🔟 Automatic metrics (NLI + Self-BLEU)

```bash
python -m src.eval.automatic_metrics \
    --input-file outputs/mascot_eval/QMSum/test/<generated-file>.jsonl \
    --nli-model FacebookAI/roberta-large-mnli \
    --pairing-mode same-round \
    --output-file outputs/mascot_eval/QMSum/test/metrics_auto.json
```

`--pairing-mode same-round` (default) pairs agent responses within a round using each turn's `round_id`; legacy files without `round_id` require `--allow-round-inference` or an explicit `--pairing-mode all-pairs` / `adjacent-agent`. `--nli-direction temporal` (default, paper definition: earlier turn = premise) or `bidirectional` for a robustness analysis.

## 📊 Datasets

Datasets are pulled from Hugging Face automatically (see `DATASET2HF` in `src/const.py`).

| Dataset                          | Domain            | Task                                                   |
| :------------------------------- | :---------------- | :----------------------------------------------------- |
| 📋**QMSum**                | Meetings          | Multi-agent meeting assistance                         |
| 💛**ESConv**               | Emotional support | Supportive multi-agent dialogue (with user simulation) |
| 💬**Empathetic Dialogues** | Emotional support | Supportive multi-agent dialogue (with user simulation) |

## 📦 Output Format

Episode generators write JSON Lines: one conversation object per line. Every record carries a schema metadata block, and each run directory additionally contains `run_config.json` (resolved startup configuration) and `run_summary.json` (failure/fallback counters, timing, status).

```jsonc
{
  "conv_id": "...",               // QMSum records use "id"/"start_idx"/"meeting" instead
  "turns": [
    {
      "turn_idx": 0,
      "round_id": 0,              // one round = one Director->Speaker(->User) cycle
      "directive": {              // parsed Director directive for this turn
        "agent_name": "...", "action": "...", "tone": "..."
      },
      "agent_name": "...",        // selected speaker
      "response": "...",          // Speaker response conditioned on the directive
      "directive_output": "..."   // raw Director model output
    }
  ],
  "agents_spoken": {"...": 1},
  "metadata": {
    "schema_version": "1.0",
    "system": "mascot",           // mascot | mascot-g | mascot-i
    "dataset": "QMSum",
    "persona_set": "default",     // or "big5"
    "director_model": "Qwen/Qwen3-8B",
    "speaker_model": "Qwen/Qwen3-8B",
    "user_model": "Qwen/Qwen3-8B",  // only present when user simulation ran
    "rubric_version": "paper_v1"
  }
}
```

Empathetic/ESConv records additionally carry `initial_emotion`, `final_emotion`, `emotion_trajectory`, `user_big_five`, and user-simulation turns (`"turn_type": "user"`). Rubric evaluation writes `metrics_*.json` files next to the input (or under `--output-dir`) and dispatches on the metadata block (legacy files without it fall back to filename heuristics with a warning).

In tolerant mode (default) generation records `num_generation_failures`, `num_directive_parse_failures`, `num_random_agent_fallbacks`, `num_empty_responses`, `num_annotation_failures`, and `num_skipped_samples` in `run_summary.json`; `--strict` terminates on the first failure instead.

## 📝 Citation

If you find MASCOT useful in your research, please cite:

```
@inproceedings{wang2026mascot,
  title={MASCOT: Towards Multi-Agent Socio-Collaborative Companion Systems},
  author={Wang, Yiyang and Jin, Yiqiao and Cabral, Alex and Hester, Josiah},
  booktitle={EMNLP},
  year={2026}
}
```

## ⚖️ License

The **source code** in this repository is released under the [Apache License 2.0](LICENSE). 