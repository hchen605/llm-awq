from torch.nn.utils import prune
import torch
import torch.nn as nn

# Apply pruning to the model
def unstructured_pruning(model, amount=0.5):
    """
    Apply global unstructured pruning to the entire model.

    Args:
        model: The PyTorch model to be pruned.
        amount: Fraction of weights to prune (default is 0.5).

    Returns:
        Pruned model.
    """
    for name, module in model.named_modules():
        # Only apply pruning to linear layers
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")  # Remove pruning to leave a dense tensor

import torch

def prune_llama_attention_heads(model, layer_indices, heads_to_prune):
    """
    Prune entire attention heads for specified layers.

    Args:
        model: A LLaMA model from Hugging Face Transformers.
        layer_indices: list of layer indices to prune (e.g., [0, 1, 2] or [0] for the first layer).
        heads_to_prune: dictionary {layer_idx: [head_indices_to_prune]}, 
                        or a single list if you want the same heads pruned across layers.
    """
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx].self_attn

        # Number of heads and hidden dimension
        hidden_size = layer.q_proj.weight.shape[0]
        num_heads = layer.num_heads  # might differ depending on your LLaMA code
        head_dim = hidden_size // num_heads

        # If heads_to_prune is a dict keyed by layer index, get the list for this layer
        if isinstance(heads_to_prune, dict):
            heads = heads_to_prune.get(layer_idx, [])
        else:
            heads = heads_to_prune  # assume same for all

        # Build a mask for which heads to keep
        heads_to_keep = set(range(num_heads)) - set(heads)
        heads_to_keep = sorted(list(heads_to_keep))

        # For Q, K, V:
        # Each is shape [hidden_size, hidden_size], but can be viewed
        # as [num_heads * head_dim, num_heads * head_dim].
        # We'll keep the rows/columns corresponding to the heads we want.
        
        def prune_proj(proj):
            # proj.weight is [hidden_size, hidden_size]
            # we interpret it as [num_heads, head_dim, num_heads, head_dim] for clarity
            W = proj.weight.data
            B = proj.bias.data if proj.bias is not None else None

            # 1) Prune rows
            W = W.view(num_heads, head_dim, hidden_size)
            W = W[heads_to_keep, :, :]  # keep the relevant heads along "rows"
            W = W.view(-1, hidden_size) 

            # 2) Prune columns if needed (for multi-query or multi-headed attention,
            #    you might also want to remove columns in Q, K, V. 
            #    But if it’s a typical design, columns are not grouped by head the same way.
            #    Usually we only need to prune the dimension that references heads explicitly.
            
            # If there's a bias, prune it similarly in the "heads" dimension
            if B is not None:
                B = B.view(num_heads, head_dim)
                B = B[heads_to_keep, :]
                B = B.view(-1)

            # Assign back
            proj.weight = torch.nn.Parameter(W)
            if B is not None:
                proj.bias = torch.nn.Parameter(B)

        prune_proj(layer.q_proj)
        prune_proj(layer.k_proj)
        prune_proj(layer.v_proj)

        # The output projection often merges heads again,
        # so you typically only need to prune the input dimension referencing heads.
        # shape [hidden_size, hidden_size] => treat input dimension (dim=1).
        def prune_output_proj(proj):
            W = proj.weight.data
            B = proj.bias.data if proj.bias is not None else None

            W = W.view(hidden_size, num_heads, head_dim)
            W = W[:, heads_to_keep, :]  # prune "head" dimension in the input
            W = W.view(hidden_size, -1)

            proj.weight = torch.nn.Parameter(W)
            if B is not None:
                # B is just shape [hidden_size], usually no direct head dimension
                proj.bias = torch.nn.Parameter(B)

        prune_output_proj(layer.o_proj)

        # Update the stored number of heads
        new_num_heads = len(heads_to_keep)
        layer.num_heads = new_num_heads

    return model

def structured_column_pruning(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
    return model

# # Example usage
# model_id = "meta-llama/Llama-3.1-8B"
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto"
# )

# # Let's say we want to prune heads [0,1,2] in the first 2 layers
# layer_indices_to_prune = [0, 1]
# heads_to_prune = {
#     0: [0, 1, 2],  # heads to prune in layer 0
#     1: [0, 1, 2]   # heads to prune in layer 1
# }
# model = prune_llama_attention_heads(model, layer_indices_to_prune, heads_to_prune)


def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    Magnitude-based pruning for a single tensor.
    This function computes a binary mask (as a float tensor of ones and zeros)
    and zeroes out the weights in-place.
    
    Args:
        tensor: The weight tensor of a layer.
        sparsity: Fraction of elements to prune (0.0 to 1.0).
    
    Returns:
        A binary mask (torch.Tensor) with 1 for kept weights and 0 for pruned.
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()
    # Step 1: Determine number of zeros to set
    num_zeros = round(num_elements * sparsity)
    # Step 2: Compute importance as the absolute value of the weights
    importance = torch.abs(tensor)
    # Step 3: Find the threshold (kth smallest value)
    threshold = torch.kthvalue(importance.flatten(), num_zeros)[0]
    # Step 4: Create the binary mask (using >= to include values equal to threshold)
    mask = (importance >= threshold).to(tensor.dtype)
    # Step 5: Zero out pruned weights in-place
    with torch.no_grad():
        tensor.mul_(mask)
        
    return mask

def apply_custom_pruning(model: nn.Module, amount: float) -> nn.Module:
    """
    Applies custom in-place magnitude-based pruning to all linear layers in the model.
    It stores a binary mask as a buffer and registers a gradient hook to enforce sparsity.
    
    Args:
        model: The neural network model.
        amount: The fraction of weights to prune in each linear layer.
    
    Returns:
        The pruned model.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Create the pruning mask for the weight tensor.
            mask = fine_grained_prune(module.weight, amount)
            # To save memory, store the mask as a boolean tensor.
            module.register_buffer("prune_mask", mask.to(torch.bool))
            
            # Define a hook to zero out gradients for pruned weights.
            def hook(grad, mask=mask):
                # Multiply gradients by the mask (converted to grad.dtype).
                return grad * mask.to(grad.dtype)
            
            # Register the hook on the weight parameter.
            module.weight.register_hook(hook)
    return model

# Example usage:
# Assume 'model' is your pre-trained LLaMA model.
# prune_amount = 0.2  # For example, prune 20% of the weights in each linear layer.
# model = apply_custom_pruning(model, amount=prune_amount)
