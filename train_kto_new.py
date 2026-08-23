from peft import LoraConfig, get_peft_model
import os
# Set the path at the very top, before any other imports
os.environ["HF_HOME"] = "/home/users/kagyemang-dua/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "/home/users/kagyemang-dua/huggingface_cache"

import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOTrainer, KTOConfig

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.bfloat16
    )

    peft_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    # 3. Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model = model.to("cuda")

    # Load Dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # KTO Configuration
    training_args = KTOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        remove_unused_columns=False,
    )

    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    train()
