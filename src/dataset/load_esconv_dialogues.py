"""
CLI wrapper for loading the ESConv dataset.

The actual loader lives in ``src.utils.data_utils.load_esconv_dialogues``.
This script just exposes it on the command line:

    python src/dataset/load_esconv_dialogues.py --split train
"""

if __name__ == "__main__":
    import argparse

    from src.const import ESCONV_DATASET, DATASET2HF
    from src.utils.data_utils import load_esconv_dialogues

    parser = argparse.ArgumentParser(description="Load ESConv dataset")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "valid", "validation", "test"])
    parser.add_argument("--dataset", type=str, default=DATASET2HF[ESCONV_DATASET])
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    rows = load_esconv_dialogues(
        split=args.split,
        dataset_name=args.dataset,
        cache_dir=args.cache_dir,
    )
    print(f"\nTotal utterances: {len(rows):,}")
    if rows:
        print(f"Columns: {list(rows[0].keys())}")
        print(f"\nSample row 0: {rows[0]}")
        first_conv = rows[0]["conv_id"]
        conv_rows = [r for r in rows if r["conv_id"] == first_conv]
        print(f"\nFirst conversation has {len(conv_rows)} turns:")
        for r in conv_rows[:6]:
            speaker = "USR" if r["speaker_idx"] == "1" else "SYS"
            print(f"  [{speaker}] {r['utterance'][:120]}")
