from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch

def prune_finetune(model, tokenizer, path):
    model.train()
    
    # Ensure a pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Define the tokenization function for causal language modeling.
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048)

    # ----- Load WikiText‑2 Dataset for Fine-Tuning -----
    print("Loading WikiText‑2 dataset for fine-tuning...")
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    # (Optional) If you want to use only a subset of the train data, you can uncomment:
    # subset_size = int(0.1 * len(train_dataset))  # for example, 10% of the data
    # train_dataset = train_dataset.shuffle(seed=42).select(range(subset_size))
    
    # Tokenize the datasets and remove the raw text column
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Data collator for causal language modeling (without masked LM)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Print GPU information
    num_devices = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_devices}", flush=True)
    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        print(f"GPU {i}: {device_name}")
        print(f"  Memory Allocated: {allocated / 1e6:.2f} MB", flush=True)
        print(f"  Memory Reserved: {reserved / 1e6:.2f} MB", flush=True)

    # ---------------------------------------------
    # Fine-Tuning with Hugging Face Trainer API
    # ---------------------------------------------
    training_args = TrainingArguments(
        output_dir=path,
        overwrite_output_dir=True,
        num_train_epochs=2,            # Increase the number of epochs for better performance
        per_device_train_batch_size=4, # Adjust based on your GPU memory
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=50,
        save_steps=200,
        save_total_limit=1,
        bf16=True,                     # Use BF16 if supported
        report_to="none",              # Set to "wandb" or "tensorboard" if needed
        load_best_model_at_end=True,   # Enable best model loading
        metric_for_best_model="eval_loss",  # Use evaluation loss as the metric
        greater_is_better=False,       # Lower eval loss is better
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # Start fine-tuning the pruned model
    trainer.train()

    # Evaluate the final model on the validation split
    eval_metrics = trainer.evaluate()
    print("Evaluation Metrics:", eval_metrics)

    # Save the final pruned and fine-tuned model if desired
    # trainer.save_model("pruned-llama-finetuned-model")
