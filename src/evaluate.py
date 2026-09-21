"""Compare the base model against the QLoRA fine-tuned adapter on the held
-out test split, and write a before/after report to results/eval_results.md.

Requires a GPU. Run after src/finetune.py has produced an adapter in
config.ADAPTER_DIR:

    python src/evaluate.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import prompting  # noqa: E402

DISCLAIMER = (
    "**Disclaimer:** this model is a small-scale fine-tuning skills demo. "
    "It has not been clinically validated and must not be used for real "
    "medical decisions."
)


def load_test_examples(n):
    examples = []
    with open(config.TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= n:
                break
    return examples


def load_models():
    """Load the base model once and attach the LoRA adapter on top.

    Important: `PeftModel.from_pretrained(base_model, ...)` mutates
    `base_model` in place (it injects LoRA layers into it), so a separately
    kept reference to the "base" model would silently run WITH the adapter
    too. We instead load a single PeftModel and toggle the adapter on/off
    via its `disable_adapter()` context manager to get true baseline vs.
    fine-tuned comparisons from one model in memory (also halves VRAM use
    versus loading two full copies on a free T4).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.BNB_LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    base_model.config.use_cache = True

    model = PeftModel.from_pretrained(base_model, config.ADAPTER_DIR)
    return model, tokenizer


def generate(model, tokenizer, question):
    import torch

    prompt = prompting.build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip the prompt's own text back off, keep only the completion.
    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return decoded[len(prompt_decoded):].strip()


def score_rouge_l(prediction, reference):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def main():
    if not os.path.exists(config.ADAPTER_DIR):
        raise FileNotFoundError(
            f"{config.ADAPTER_DIR} not found. Run `python src/finetune.py` first."
        )

    examples = load_test_examples(config.EVAL_NUM_EXAMPLES)
    print(f"Evaluating on {len(examples)} held-out questions")

    model, tokenizer = load_models()

    rows = []
    base_scores, ft_scores = [], []

    for i, ex in enumerate(examples):
        question, reference = ex["question"], ex["answer"]
        with model.disable_adapter():
            base_answer = generate(model, tokenizer, question)
        ft_answer = generate(model, tokenizer, question)

        base_score = score_rouge_l(base_answer, reference)
        ft_score = score_rouge_l(ft_answer, reference)
        base_scores.append(base_score)
        ft_scores.append(ft_score)

        rows.append({
            "question": question,
            "reference": reference,
            "base_answer": base_answer,
            "finetuned_answer": ft_answer,
            "base_rouge_l": round(base_score, 3),
            "finetuned_rouge_l": round(ft_score, 3),
        })
        print(f"[{i + 1}/{len(examples)}] base ROUGE-L={base_score:.3f} | finetuned ROUGE-L={ft_score:.3f}")

    avg_base = sum(base_scores) / len(base_scores)
    avg_ft = sum(ft_scores) / len(ft_scores)

    write_report(rows, avg_base, avg_ft)


def write_report(rows, avg_base, avg_ft):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "eval_results.md")

    lines = [
        "# Evaluation: base model vs. QLoRA fine-tuned model",
        "",
        DISCLAIMER,
        "",
        f"- Base model: `{config.BASE_MODEL_ID}` (4-bit, no adapter)",
        f"- Fine-tuned model: `{config.BASE_MODEL_ID}` + LoRA adapter "
        f"(r={config.LORA_R}, alpha={config.LORA_ALPHA}) trained on "
        f"{config.RAW_DATASET_ID}",
        f"- Metric: ROUGE-L F1 against the reference answer, {len(rows)} "
        "held-out test questions",
        "",
        f"| | Average ROUGE-L |",
        f"|---|---|",
        f"| Base model | {avg_base:.3f} |",
        f"| Fine-tuned model | {avg_ft:.3f} |",
        "",
        "## Qualitative examples",
        "",
    ]

    for i, row in enumerate(rows[:10]):
        lines += [
            f"### Example {i + 1}",
            f"**Question:** {row['question']}",
            "",
            f"**Reference answer:** {row['reference']}",
            "",
            f"**Base model (ROUGE-L={row['base_rouge_l']}):** {row['base_answer']}",
            "",
            f"**Fine-tuned model (ROUGE-L={row['finetuned_rouge_l']}):** {row['finetuned_answer']}",
            "",
            "---",
            "",
        ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    raw_path = os.path.join(config.RESULTS_DIR, "eval_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nWrote report -> {path}")
    print(f"Wrote raw results -> {raw_path}")


if __name__ == "__main__":
    main()
