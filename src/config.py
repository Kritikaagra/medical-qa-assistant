"""Central configuration for the medical Q&A QLoRA fine-tuning project.

Every other script (prepare_dataset.py, finetune.py, evaluate.py, inference.py)
imports from here so hyperparameters live in exactly one place.
"""

import os

# ---------------------------------------------------------------------------
# Model & dataset
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"  # ungated, fits 4-bit on a T4
RAW_DATASET_ID = "keivalya/MedQuad-MedicalQnADataset"  # NIH-sourced medical Q&A pairs

# Candidate (question, answer) column names across dataset variants. The prep
# script auto-detects whichever pair is present instead of hard-coding one.
QUESTION_COLUMN_CANDIDATES = ["Question", "question", "input", "instruction"]
ANSWER_COLUMN_CANDIDATES = ["Answer", "answer", "output", "response"]

SYSTEM_INSTRUCTION = (
    "You are a medical information assistant. Answer the following health "
    "question clearly and accurately for a general audience. This is "
    "educational information only, not a substitute for professional "
    "medical advice."
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "adapter")  # trained LoRA adapter output

TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.jsonl")
VAL_FILE = os.path.join(PROCESSED_DIR, "val.jsonl")
TEST_FILE = os.path.join(PROCESSED_DIR, "test.jsonl")

# ---------------------------------------------------------------------------
# Dataset sizing (kept small on purpose: free-tier Colab T4 has a session
# time limit, and this is a skills demo, not a production model)
# ---------------------------------------------------------------------------
MAX_TRAIN_EXAMPLES = 3000
MAX_VAL_EXAMPLES = 300
MAX_TEST_EXAMPLES = 200
MIN_ANSWER_CHARS = 20     # drop near-empty answers
MAX_ANSWER_CHARS = 1500   # drop outlier essay-length answers to control seq length
SEED = 42

# ---------------------------------------------------------------------------
# 4-bit quantization (BitsAndBytesConfig)
# ---------------------------------------------------------------------------
BNB_LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"        # normal-float 4-bit, per the QLoRA paper
BNB_4BIT_USE_DOUBLE_QUANT = True   # quantize the quantization constants too
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"

# ---------------------------------------------------------------------------
# LoRA (PEFT)
# ---------------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = 512
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.001
MAX_GRAD_NORM = 0.3
LOGGING_STEPS = 10
SAVE_STEPS = 100
OPTIM = "paged_adamw_32bit"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_NUM_EXAMPLES = 20       # how many held-out test questions to score/report
EVAL_MAX_NEW_TOKENS = 200
