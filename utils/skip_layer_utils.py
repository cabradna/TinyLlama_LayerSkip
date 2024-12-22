import torch
import math

def should_exit_early(logits, layer_index, current_step, exit_threshold=0.9, total_steps=10000, num_layers=32):
    """
    Determines whether the model should exit early at a given layer during training,
    based on the current training step and the maximum predicted probability.

    Args:
        logits (torch.Tensor): The output logits from the early exit branch.
        layer_index (int): The index of the current layer.
        current_step (int): The current training step.
        exit_threshold (float, optional): The threshold probability for early exit. Defaults to 0.9.
        total_steps (int, optional): The total number of training steps. Defaults to 10000.
        num_layers (int, optional): The total number of layers in the model. Defaults to 32.

    Returns:
        bool: True if the model should exit early, False otherwise.
    """
    # Get probabilities from logits
    probs = torch.softmax(logits, dim=-1)
    max_prob = torch.max(probs, dim=-1)[0].max()  # Get maximum probability
    
    # Calculate curriculum progress
    enable_layer_step = total_steps // (2 * num_layers)
    enabled_layer = current_step // enable_layer_step
    
    # Exit if confidence is high enough and layer is enabled
    if max_prob >= exit_threshold and layer_index >= (num_layers - 1 - enabled_layer):
        return True
    return False

def calculate_layer_scale(layer_idx: int, total_layers: int) -> float:
    """
    Calculate D(l) - the per-layer scaling function
    Using equation (3): D(l) = e^(l*ln2/(L-1)) - 1
    
    Args:
        layer_idx (int): Current layer index l
        total_layers (int): Total number of layers L
    """
    if layer_idx == 0 or total_layers <= 1:
        return 0.0  # No dropout for first layer
    return math.exp((layer_idx * math.log(2)) / (total_layers - 1)) - 1

def calculate_dropout_rate(
    layer_idx: int, 
    total_layers: int,
    p_max: float = 0.8  # hyperparameter for maximum dropout rate
) -> float:
    """
    Calculate p_l - the dropout rate for layer l
    Using simplified equation: p_l = D(l)p_max
    
    Args:
        layer_idx (int): Current layer index l
        total_layers (int): Total number of layers L
        p_max (float): Maximum dropout rate hyperparameter
    """
    D_l = calculate_layer_scale(layer_idx, total_layers)
    return min(max(D_l * p_max, 0.0), 1.0)  # Clamp between 0 and 1

def calculate_curriculum_C(current_step, layer_index, num_layers, total_steps):
    """
    Implements the binary curriculum function C(t,l) that determines if early exit 
    is enabled for a given layer at the current training step.
    
    Args:
        current_step (int): Current training step
        layer_index (int): Index of the current layer (0-indexed)
        num_layers (int): Total number of layers in the model
        total_steps (int): Total number of training steps
    
    Returns:
        float: 1.0 if early exit is enabled for this layer, 0.0 otherwise
    """
    # Calculate the interval for enabling new layers (T/2L in the paper)
    enable_layer_step = total_steps // (2 * num_layers)
    
    # Determine which layer is currently enabled (counting from the last layer)
    enabled_layer = current_step // enable_layer_step
    
    # Return 1.0 if the layer is enabled, 0.0 otherwise
    return int(1.0) if layer_index >= (num_layers - 1 - enabled_layer) else int(0.0)


def calculate_layer_scale_e(layer_index, num_layers, e_scale=0.5):
    """
    Calculates the layer-wise scaling factor e(l) that gives higher weights 
    to later layers.
    
    Args:
        layer_index (int): Index of the current layer (0-indexed)
        num_layers (int): Total number of layers in the model
        e_scale (float): Hyperparameter that controls the scale (0 ≤ e_scale ≤ 1)
    
    Returns:
        float: The scaling factor for this layer
    """
    if layer_index < num_layers - 1:
        # For all layers except the last: e_scale * sum(i from 0 to l)
        return e_scale * sum(i for i in range(layer_index + 1))
    else:
        # For the last layer: (L-1 + e_scale * sum(i from 0 to L-2))
        return (num_layers - 1 + e_scale * sum(i for i in range(num_layers - 1)))


def calculate_early_exit_scale_factor(current_step, layer_index, num_layers, total_steps, e_scale=0.5):
    """
    Calculates the normalized per-layer loss scale ẽ(t,l) combining both the curriculum
    and the layer-wise scaling.
    
    Args:
        current_step (int): Current training step
        layer_index (int): Index of the current layer (0-indexed)
        num_layers (int): Total number of layers in the model
        total_steps (int): Total number of training steps
        e_scale (float): Hyperparameter that controls the scale (0 ≤ e_scale ≤ 1)
    
    Returns:
        float: Normalized scaling factor for the early exit loss
    """
    # Get curriculum value for current layer
    C_t_l = calculate_curriculum_C(current_step, layer_index, num_layers, total_steps)
    
    # Get layer scale for current layer
    e_l = calculate_layer_scale_e(layer_index, num_layers, e_scale)
    
    # Calculate denominator: sum of (C(t,i) * e(i)) for all layers
    denominator = sum(
        calculate_curriculum_C(current_step, i, num_layers, total_steps) * 
        calculate_layer_scale_e(i, num_layers, e_scale)
        for i in range(num_layers)
    )
    
    # Calculate final normalized scale factor
    if denominator > 0:
        return (C_t_l * e_l) / denominator
    return 0.0

def evaluate(model, val_dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids)
                loss = criterion(outputs, targets)
            total_loss += loss.item() 
    
    return total_loss / len(val_dataloader)