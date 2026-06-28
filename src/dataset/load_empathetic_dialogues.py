"""
CLI wrapper for loading the EmpatheticDialogues dataset.

The actual loader lives in ``src.utils.data_utils.load_empathetic_dialogues``.
This script just exposes it on the command line:

    python src/dataset/load_empathetic_dialogues.py --split train
"""

if __name__ == '__main__':
    import argparse

    from src.const import EMPATHETIC_DATASET, DATASET2HF
    from src.utils.data_utils import load_empathetic_dialogues

    parser = argparse.ArgumentParser(
        description='Load EmpatheticDialogues dataset from Hugging Face or local files'
    )
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'valid', 'validation', 'test'],
                       help='Dataset split to load')
    parser.add_argument('--dataset', type=str, default=DATASET2HF[EMPATHETIC_DATASET],
                       help=f'HuggingFace dataset name (default: {DATASET2HF[EMPATHETIC_DATASET]})')
    parser.add_argument('--local-path', type=str, default=None,
                       help='Local path to dataset directory (expects CSV files named {split}.csv)')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Cache directory for downloaded datasets')
    args = parser.parse_args()

    print("=" * 60)
    print("EmpatheticDialogues Dataset Loader")
    print("=" * 60)

    try:
        train_data = load_empathetic_dialogues(
            split=args.split,
            dataset_name=args.dataset,
            local_path=args.local_path,
            cache_dir=args.cache_dir,
        )

        print(f'\n{"=" * 60}')
        print('Dataset Information')
        print('=' * 60)
        print(f'Dataset columns: {list(train_data[0].keys())}')
        print(f'Total utterances: {len(train_data):,}')

        print(f'\n{"=" * 60}')
        print('Sample Utterances')
        print('=' * 60)
        for i in range(min(3, len(train_data))):
            print(f'\n{i + 1}. Conversation: {train_data[i].get("conv_id", "N/A")}')
            print(f'   Emotion: {train_data[i].get("context", train_data[i].get("emotion", "N/A"))}')

            if "speaker_idx" in train_data[i]:
                speaker_idx = str(train_data[i]["speaker_idx"])
                print(f'   Speaker: {"Speaker" if speaker_idx == "1" else "Listener"}')

            if "utterance" in train_data[i]:
                utterance_text = train_data[i]["utterance"]
                print(f'   Utterance: {utterance_text[:100]}{"..." if len(utterance_text) > 100 else ""}')

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
