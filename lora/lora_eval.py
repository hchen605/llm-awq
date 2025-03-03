from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import tqdm
import argparse
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Causal LM")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the pre-trained model checkpoint",
)
parser.add_argument(
    "--lora_path",
    type=str,
    required=True,
    help="Path where the fine-tuned model and tokenizer will be saved",
)
parser.add_argument(
    "--prune_amount",
    type=float,
    default=0.3,
    help="Fraction of weights to prune in each Linear layer (default: 0.3)",
)
args = parser.parse_args()

def evaluate(model, tokenizer, nsamples=40, seq_length=2048):
    # Load the Wikitext-2 test dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
    # Tokenize the dataset and handle large inputs
    tokenized_text = tokenizer("\n\n".join(dataset['text']), return_tensors='pt', truncation=False)
    input_ids = tokenized_text['input_ids'].to(model.device)
    nsamples = input_ids.numel() // 2048
    # Ensure model is in evaluation mode
    model.eval()

    # Initialize variables to store losses
    nlls = []

    # Iterate over the dataset in chunks
    for i in tqdm.tqdm(range(nsamples), desc="Evaluating..."):
        # Determine start and end indices for the batch
        start_idx = i * seq_length
        end_idx = (i + 1) * seq_length

        if start_idx >= input_ids.size(1):
            break  # End of dataset

        # Slice the input_ids for this batch
        batch = input_ids[:, start_idx:end_idx]
        
        # Skip if the batch size is smaller than seq_length
        if batch.size(1) < seq_length:
            continue

        # Compute loss for the batch
        with torch.no_grad():
            outputs = model(batch, labels=batch)
            loss = outputs.loss  # Cross-entropy loss for the batch
        
        # Convert to negative log-likelihood and scale by sequence length
        neg_log_likelihood = loss * batch.size(1)
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    total_nll = torch.stack(nlls).sum()
    total_tokens = nsamples * seq_length
    perplexity = torch.exp(total_nll / total_tokens)

    return perplexity.item()

def evaluate_2(model, tokenizer, nsamples=40, seq_length=2048):
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = 2048
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

# Define the paths (adjust as needed)
base_model_path = args.model_path
lora_model_path = args.lora_path

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)

print(f"Pruning linear layers with {args.prune_amount*100:.1f}% sparsity...")
for name, module in base_model.named_modules():
    if isinstance(module, nn.Linear):
        # Apply L1 unstructured pruning on the weight parameter
        prune.l1_unstructured(module, name="weight", amount=args.prune_amount)
        prune.remove(module, "weight")

# Wrap the base model with the LoRA adapter weights
model = PeftModel.from_pretrained(base_model, lora_model_path)

# Set the model to evaluation mode
model.eval()
model.to(device)

perplexity = evaluate(model, tokenizer)
print(f"Perplexity: {perplexity:.2f}")


