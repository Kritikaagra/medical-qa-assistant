"""Single source of truth for the Mistral-7B-Instruct-v0.2 prompt template.

Used by prepare_dataset.py (training text), evaluate.py and inference.py
(generation prompt) so the format can never drift between training and
inference.
"""

import config


def build_instruction(question):
    return f"{config.SYSTEM_INSTRUCTION}\n\nQuestion: {question}"


def build_prompt(question):
    """Prompt used at inference time (no answer, model completes it)."""
    return f"<s>[INST] {build_instruction(question)} [/INST]"


def build_training_text(question, answer):
    """Full labeled example used for supervised fine-tuning."""
    return f"{build_prompt(question)} {answer}</s>"
