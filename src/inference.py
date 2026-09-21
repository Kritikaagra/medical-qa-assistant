"""Chat with the fine-tuned medical Q&A adapter.

Requires a GPU and a trained adapter in config.ADAPTER_DIR (see
src/finetune.py).

Usage:
    python src/inference.py --question "What are the symptoms of anemia?"
    python src/inference.py            # interactive loop, empty line to quit
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import prompting  # noqa: E402

DISCLAIMER = (
    "This is a fine-tuning skills demo, not a clinically validated medical "
    "tool. Do not use it for real medical decisions.\n"
)


def load_model_and_tokenizer():
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
    model.eval()
    return model, tokenizer


def ask(model, tokenizer, question, max_new_tokens=config.EVAL_MAX_NEW_TOKENS):
    import torch

    prompt = prompting.build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return decoded[len(prompt_decoded):].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=None, help="Ask a single question and exit.")
    parser.add_argument("--max-new-tokens", type=int, default=config.EVAL_MAX_NEW_TOKENS)
    args = parser.parse_args()

    if not os.path.exists(config.ADAPTER_DIR):
        raise FileNotFoundError(
            f"{config.ADAPTER_DIR} not found. Run `python src/finetune.py` first."
        )

    print(DISCLAIMER)
    model, tokenizer = load_model_and_tokenizer()

    if args.question:
        print(ask(model, tokenizer, args.question, args.max_new_tokens))
        return

    print("Interactive mode — type a question, empty line to quit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            break
        answer = ask(model, tokenizer, question, args.max_new_tokens)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
