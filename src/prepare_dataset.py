"""Download the raw medical Q&A dataset, clean it, format it into Mistral's
instruction template, and write train/val/test JSONL splits.

Run:
    python src/prepare_dataset.py

Designed to run on any machine with internet access (no GPU needed) so you
can prepare data locally and only spend Colab GPU time on src/finetune.py.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import prompting  # noqa: E402


def detect_column(columns, candidates, role):
    for name in candidates:
        if name in columns:
            return name
    raise ValueError(
        f"Could not find a {role} column in dataset columns {columns}. "
        f"Tried: {candidates}. Pass --question-col/--answer-col explicitly, "
        f"or update QUESTION_COLUMN_CANDIDATES/ANSWER_COLUMN_CANDIDATES in "
        f"src/config.py once you've inspected the real schema."
    )


def clean_text(text):
    if text is None:
        return ""
    return " ".join(str(text).split()).strip()


def format_example(question, answer):
    """Wrap a (question, answer) pair into the training text + metadata.

    The actual template lives in prompting.py — shared with evaluate.py and
    inference.py so training and inference can never drift out of sync.
    """
    text = prompting.build_training_text(question, answer)
    return {"text": text, "question": question, "answer": answer}


def load_raw_examples(dataset_id, question_col=None, answer_col=None):
    from datasets import load_dataset

    raw = load_dataset(dataset_id)
    split_name = "train" if "train" in raw else list(raw.keys())[0]
    ds = raw[split_name]
    columns = ds.column_names
    print(f"Loaded '{dataset_id}' split='{split_name}' with columns: {columns}")

    q_col = question_col or detect_column(columns, config.QUESTION_COLUMN_CANDIDATES, "question")
    a_col = answer_col or detect_column(columns, config.ANSWER_COLUMN_CANDIDATES, "answer")
    print(f"Using question column='{q_col}', answer column='{a_col}'")

    examples = []
    for row in ds:
        q = clean_text(row.get(q_col))
        a = clean_text(row.get(a_col))
        examples.append((q, a))
    return examples


def clean_and_dedup(examples):
    seen = set()
    cleaned = []
    for q, a in examples:
        if not q or not a:
            continue
        if len(a) < config.MIN_ANSWER_CHARS or len(a) > config.MAX_ANSWER_CHARS:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((q, a))
    return cleaned


def split_examples(examples, rng):
    rng.shuffle(examples)
    n_train = min(config.MAX_TRAIN_EXAMPLES, max(0, len(examples) - config.MAX_VAL_EXAMPLES - config.MAX_TEST_EXAMPLES))
    n_val = min(config.MAX_VAL_EXAMPLES, len(examples) - n_train)
    n_test = min(config.MAX_TEST_EXAMPLES, len(examples) - n_train - n_val)

    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def write_jsonl(path, examples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q, a in examples:
            f.write(json.dumps(format_example(q, a), ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=config.RAW_DATASET_ID)
    parser.add_argument("--question-col", default=None)
    parser.add_argument("--answer-col", default=None)
    args = parser.parse_args()

    rng = random.Random(config.SEED)

    raw_examples = load_raw_examples(args.dataset_id, args.question_col, args.answer_col)
    print(f"Raw examples: {len(raw_examples)}")

    cleaned = clean_and_dedup(raw_examples)
    print(f"After cleaning/dedup: {len(cleaned)}")
    if len(cleaned) < 50:
        raise RuntimeError(
            "Fewer than 50 usable examples survived cleaning — the column "
            "auto-detection likely picked the wrong fields. Inspect the "
            "printed columns above and pass --question-col/--answer-col."
        )

    train, val, test = split_examples(cleaned, rng)
    print(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    write_jsonl(config.TRAIN_FILE, train)
    write_jsonl(config.VAL_FILE, val)
    write_jsonl(config.TEST_FILE, test)

    print("\nSample formatted training example:\n")
    print(format_example(*train[0])["text"])


if __name__ == "__main__":
    main()
