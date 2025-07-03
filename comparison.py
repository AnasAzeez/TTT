import os
import glob
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1) Configuration
MODEL_NAME = "epfl-llm/meditron-70b"
INPUT_GLOB = "/kaggle/input/main-batches-emnlp/main batches/batch_*.csv"
OUTPUT_DIR = "/kaggle/working/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2) Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
    device_map="auto" if DEVICE=="cuda" else None,
)
model.eval()

def answer_true_false(question: str, max_length: int = 32) -> str:
    # Stronger instruction and explicit format
    prompt = (
        "You are an assistant that answers ONLY True or False—no extra text.\n"
        f"Question: {question}\n"
        "Answer (one word, True or False):"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    rl = reply.lower()
    if "true" in rl.split():
        return "True"
    elif "false" in rl.split():
        return "False"
    # fallback: look anywhere
    if "true" in rl:
        return "True"
    if "false" in rl:
        return "False"
    return "Unknown"

# 3) Process each batch file with tqdm
for batch_path in tqdm(glob.glob(INPUT_GLOB), desc="Batches", unit="file"):
    df = pd.read_csv(batch_path)
    df = df.drop(columns=["answer"], errors="ignore")

    preds = []
    for q in tqdm(df["question"], desc="Questions", leave=False):
        preds.append(answer_true_false(q))

    df["predicted_answer"] = preds

    out_name = f"pred_{os.path.basename(batch_path)}"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    df.to_csv(out_path, index=False) 

print(f"Predictions in: {OUTPUT_DIR}")
