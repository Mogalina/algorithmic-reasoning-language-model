import torch
import sqlite3
from tqdm import tqdm
from decouple import config
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
from tutor.solver_rewards import compute_solver_reward, extract_code

# 1. Configuration
MODEL_ID = "google/gemma-2-2b-it"
ADAPTER_PATH = config("GEMMA_ADAPTER_PATH", "./gemma2-dsa-qlora copy")
HF_TOKEN = config("HF_TOKEN", None)

dpo_config = DPOConfig(
    output_dir=f"{ADAPTER_PATH}_dpo",
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=100,
    remove_unused_columns=False,
    bf16=False, # Use float16 for Mac compatibility
    fp16=True,
)

def fetch_prompts_from_db():
    """Fetches real problem descriptions from the database."""
    db_path = "data/app.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT description FROM interview_problems LIMIT 10")
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows if row[0]]
    except Exception as e:
        print(f"[DPO] Could not load prompts from DB: {e}")
        return ["Write a Python function to reverse a linked list."]

def train_dpo():
    """
    Performs DPO fine-tuning using compilation success as the preference signal.
    """
    print(f"[DPO] Loading model and adapter...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # 2. Prepare Dataset (On-the-fly generation of pairs)
    prompts = fetch_prompts_from_db()
    train_dataset = []

    print("[DPO] Generating preference pairs (this may take a moment)...")
    for prompt in tqdm(prompts):
        # Generate two candidate responses to compare
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.9, num_return_sequences=2)
        
        resp1 = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        resp2 = tokenizer.decode(outputs[1][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Check compilation for both
        reward1 = compute_solver_reward(resp1)
        reward2 = compute_solver_reward(resp2)

        # We create a pair only if they differ in quality
        if reward1["compiled"] and not reward2["compiled"]:
            train_dataset.append({
                "prompt": prompt,
                "chosen": resp1,
                "rejected": resp2
            })
        elif reward2["compiled"] and not reward1["compiled"]:
            train_dataset.append({
                "prompt": prompt,
                "chosen": resp2,
                "rejected": resp1
            })

    if not train_dataset:
        print("[DPO] No preference pairs found (all generated code compiled or failed equally). Try increasing temperature.")
        return

    print(f"[DPO] Training on {len(train_dataset)} pairs...")

    # 3. Initialize DPOTrainer
    dpo_trainer = DPOTrainer(
        model,
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        beta=0.1, # Strength of DPO penalty
    )

    dpo_trainer.train()
    
    print(f"[DPO] Saving improved adapter to {ADAPTER_PATH}_dpo")
    model.save_pretrained(f"{ADAPTER_PATH}_dpo")
    print("[DPO] Done!")

if __name__ == "__main__":
    train_dpo()
