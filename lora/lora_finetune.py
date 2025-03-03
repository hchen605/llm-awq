import argparse
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType

def lora_finetune(model_path, output_path):
    # Load the model and tokenizer from the provided model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.train()
    
    # Ensure a pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Setup LoRA configuration ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8,                     # Rank of LoRA matrices; adjust as needed
        lora_alpha=32,           # Scaling factor
        lora_dropout=0.1         # Dropout applied to LoRA layers
    )
    # Wrap the model with LoRA
    model = get_peft_model(model, lora_config)
    print("LoRA model parameters:")
    model.print_trainable_parameters()

    # --- Define the tokenization function for causal language modeling ---
    def tokenize_function(examples):
        # Adjust the max_length as needed (e.g., max_length=128 or 2048)
        return tokenizer(examples["text"], truncation=True, max_length=2048)

    # --- Load WikiText-2 Dataset for Fine-Tuning ---
    print("Loading WikiText-2 dataset for fine-tuning...")
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    # (Optional) Use a subset for quick experiments:
    # train_dataset = train_dataset.shuffle(seed=42).select(range(1000))
    # eval_dataset = eval_dataset.shuffle(seed=42).select(range(200))
    
    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Data collator for causal language modeling (without masked LM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Print GPU information
    num_devices = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_devices}")
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        print(f"GPU {i}: {device_name}")
        print(f"  Memory Allocated: {allocated / 1e6:.2f} MB")
        print(f"  Memory Reserved: {reserved / 1e6:.2f} MB")

    # --- Setup TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=2,            # Increase epochs if needed
        per_device_train_batch_size=4, # Adjust based on GPU memory
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=50,
        save_steps=200,
        save_total_limit=1,
        bf16=True,                     # Use BF16 if supported by your hardware
        report_to="none",              # Set to "wandb" or "tensorboard" if needed
        load_best_model_at_end=True,   # Enable best model loading
        metric_for_best_model="eval_loss",  # Use evaluation loss as metric
        greater_is_better=False,       # Lower eval loss is better
        max_grad_norm=1.0              # Enable gradient clipping for stability
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # --- Fine-tune with LoRA ---
    trainer.train()

    # Evaluate the final model on the validation split
    eval_metrics = trainer.evaluate()
    print("Evaluation Metrics:", eval_metrics)

    # Optionally, save the LoRA-adapted model and tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Causal LM")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the fine-tuned model and tokenizer will be saved",
    )
    
    args = parser.parse_args()
    lora_finetune(args.model_path, args.output_path)
