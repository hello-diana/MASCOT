"""Preference-data generation package (Phase-2 split of
src/dataset/generate_and_annotate.py).

Modules:
- common: pure shared helpers (episode context, GRPO columns, score scaling,
  artifact-path resolution, response cleaning).
- annotation: judge invocation (individual/group/branching) + LLM pair
  verification.
- preference_pairs: preference-pair construction for the four distinct cases
  (individual response pairs, independent group-episode pairs, branching
  directive pairs, legacy two-run conversion).
- serialization: JSONL/HF-Dataset writing, output policy (refuse/resume/
  overwrite), per-run sidecar metadata.
- individual / group_independent / group_branching: the three pipelines.

The documented CLI remains ``python -m src.dataset.generate_and_annotate``,
which re-exports every public name from here.
"""
