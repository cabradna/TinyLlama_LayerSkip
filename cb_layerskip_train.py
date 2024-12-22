import time
import torch
from transformers import AutoModelForCausalLM
from functools import partial
import random
import gc

from utils import cb_fused_cross_entropy as cbloss
from utils.skip_layer_utils import (
    should_exit_early,
    calculate_dropout_rate,
    calculate_curriculum_C,
    calculate_early_exit_scale_factor
)

def train_with_gradient_accumulation(
    model,
    train_dataloader,
    optimizer,
    num_epochs,
    device,
    gradient_accumulation_steps=32,
    max_grad_norm=1.0,
    scheduler=None,
    val_dataloader=None,
    eval_every=100,
    scaler=None,
):
    """
    Train a model using gradient accumulation and early exit strategy.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        optimizer: The optimizer to use
        num_epochs: Number of epochs to train
        device: Device to train on
        gradient_accumulation_steps: Number of steps to accumulate gradients
        micro_batch_size: Size of each micro batch
        max_grad_norm: Maximum gradient norm for clipping
        scheduler: Learning rate scheduler (optional)
        val_dataloader: DataLoader for validation data (optional)
        eval_every: How often to evaluate (in steps)
        scaler: Gradient scaler for mixed precision training (optional)
    """
    # Start timing for total training
    training_start_time = time.time()
    total_tokens_processed = 0
    batch_times = []
    best_val_loss = float('inf')  # Initialize best_val_loss here
    
    model.train()
    model.gradient_checkpointing_enable()  # Add at start of training
    total_steps = len(train_dataloader) * num_epochs
    current_step = 0
    accumulated_loss = 0

    # Initialize loss function
    criterion = cbloss.FusedCrossEntropyLoss().to(device)

    # Initialize gradient scaler for mixed precision training
    if scaler is None:
        scaler = torch.amp.GradScaler()

    # Initialize logging statistics
    layer_exit_counts = torch.zeros(len(model.model.layers), device=device)
    layer_losses = torch.zeros(len(model.model.layers), device=device)
    layer_loss_counts = torch.zeros(len(model.model.layers), device=device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero gradients at start of epoch
        accumulated_loss = 0
        epoch_start_time = time.time()
        epoch_tokens = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            batch_start_time = time.time()
            
            # Calculate tokens in this batch
            batch_tokens = batch['input_ids'].numel()
            total_tokens_processed += batch_tokens
            epoch_tokens += batch_tokens
            
            # Calculate if we should accumulate or update
            is_accumulating = ((batch_idx + 1) % gradient_accumulation_steps != 0) or (batch_idx + 1 == len(train_dataloader))
            
            # More aggressive memory management
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                # Force garbage collection
                gc.collect()
            
            # Move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Forward pass with mixed precision and early exit
            with torch.amp.autocast("cuda"):
                final_loss = 0
                total_scale_factor = 0
                
                # First apply token embeddings
                hidden_states = model.model.embed_tokens(input_ids)
                
                # Initialize position embeddings
                batch_size, sequence_length = input_ids.shape
                position_ids = torch.arange(0, sequence_length, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                
                # Get rotary embeddings - pass both hidden_states and position_ids
                cos, sin = model.model.rotary_emb(hidden_states, position_ids)

                # Process through layers
                for i, layer in enumerate(model.model.layers):
                    # Calculate dropout rate
                    dropout_rate = calculate_dropout_rate(
                        layer_idx=i,
                        total_layers=len(model.model.layers)
                    )
                    
                    # Create dropout mask
                    keep_mask = torch.bernoulli(
                        torch.ones(batch_size, device=device) * (1 - dropout_rate)
                    ).bool()
                    
                    if keep_mask.any():
                        if keep_mask.all():  # Process all samples
                            hidden_states = layer.input_layernorm(hidden_states)
                            
                            # Debug print
                            
                            # Prepare attention mask - Using contiguous before reshape
                            if attention_mask is not None:
                                attention_mask_4d = attention_mask.contiguous().reshape(batch_size, 1, 1, sequence_length)
                                attention_mask_4d = attention_mask_4d.to(dtype=hidden_states.dtype)
                                attention_mask_4d = (1.0 - attention_mask_4d) * torch.finfo(hidden_states.dtype).min
                            else:
                                attention_mask_4d = None
                            
                            # Attention forward pass with rotary embeddings
                            attn_outputs = layer.self_attn(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask_4d,
                                position_ids=position_ids,
                                past_key_value=None,
                                output_attentions=False,
                                use_cache=False,
                                position_embeddings=(cos, sin)  # Pass rotary embeddings
                            )
                            
                            # Process attention output
                            attn_output = attn_outputs[0]
                            hidden_states = layer.post_attention_layernorm(attn_output)
                            hidden_states = layer.mlp(hidden_states)
                            
                        else:  # Process subset of samples
                            selected_indices = torch.where(keep_mask)[0]
                            hidden_states_keep = hidden_states[selected_indices]
                            attention_mask_keep = attention_mask[selected_indices] if attention_mask is not None else None
                            position_ids_keep = position_ids[selected_indices]
                            
                            # Get rotary embeddings for the subset
                            cos_keep = cos[selected_indices] if cos is not None else None
                            sin_keep = sin[selected_indices] if sin is not None else None
                            
                            # Apply input layernorm
                            hidden_states_keep = layer.input_layernorm(hidden_states_keep)
                            
                            # Prepare attention mask for subset
                            if attention_mask_keep is not None:
                                subset_size = len(selected_indices)
                                attention_mask_4d = attention_mask_keep.contiguous().reshape(subset_size, 1, 1, sequence_length)
                                attention_mask_4d = attention_mask_4d.to(dtype=hidden_states_keep.dtype)
                                attention_mask_4d = (1.0 - attention_mask_4d) * torch.finfo(hidden_states_keep.dtype).min
                            else:
                                attention_mask_4d = None
                            
                            attn_outputs = layer.self_attn(
                                hidden_states=hidden_states_keep,
                                attention_mask=attention_mask_4d,
                                position_ids=position_ids_keep,
                                past_key_value=None,
                                output_attentions=False,
                                use_cache=False,
                                position_embeddings=(cos_keep, sin_keep)
                            )
                            
                            # Process attention output
                            attn_output = attn_outputs[0].contiguous()  # Make sure attention output is contiguous
                            hidden_states_keep = layer.post_attention_layernorm(attn_output)
                            
                            # Make sure input to MLP is contiguous and properly shaped
                            hidden_states_keep = hidden_states_keep.contiguous()
                            hidden_states_keep = layer.mlp(hidden_states_keep)
                            
                            # Update original hidden states
                            hidden_states[selected_indices] = hidden_states_keep

                    # Check for early exit
                    should_exit = calculate_curriculum_C(current_step, i, len(model.model.layers), total_steps)
                    if should_exit == 1:
                        hidden_states = hidden_states.contiguous()
                        normalized_states = model.model.norm(hidden_states)
                        current_logits = model.lm_head(normalized_states.contiguous())
                        
                        # Calculate loss immediately
                        shift_logits = current_logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        early_exit_loss = criterion(
                            shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1)
                        )
                        
                        # Calculate and apply scale factor immediately
                        scale_factor = calculate_early_exit_scale_factor(
                            current_step, i, len(model.model.layers), total_steps
                        )
                        final_loss += scale_factor * early_exit_loss
                        total_scale_factor += scale_factor
                        
                        # Update statistics
                        layer_exit_counts[i] += 1
                        layer_losses[i] += early_exit_loss.item()
                        layer_loss_counts[i] += 1
                        
                        # Clean up intermediate tensors
                        del normalized_states, current_logits, shift_logits, early_exit_loss

                # Normalize the final loss
                if total_scale_factor > 0:
                    final_loss = final_loss / gradient_accumulation_steps
                else:
                    # Handle case where no early exits occurred
                    hidden_states = model.model.norm(hidden_states)
                    logits = model.lm_head(hidden_states)
                    final_loss = criterion(
                        logits[:, :-1, :].contiguous(), 
                        input_ids[:, 1:].contiguous()
                    ) / gradient_accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(final_loss).backward()
            accumulated_loss += final_loss.item()

            # Update weights if gradient accumulation is complete
            if not is_accumulating:
                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                
                if scheduler is not None:
                    scheduler.step()
                
                optimizer.zero_grad()  # Zero gradients after scheduler step

                # Free up memory
                del final_loss
                torch.cuda.empty_cache()

                # Enhanced logging
                avg_loss = accumulated_loss / gradient_accumulation_steps
                accumulated_loss = 0

                # Validation
                if (batch_idx + 1) % (gradient_accumulation_steps * eval_every) == 0:
                # if (current_step - gradient_accumulation_steps) // eval_every > 0 and current_step // eval_every >= (current_step - gradient_accumulation_steps) // eval_every:
                    print(f'\nRunning validation at step {current_step}...')
                    eval_metrics = evaluate(model, val_dataloader, device)
                    print(f"Validation metrics:")
                    print(f"- Loss: {eval_metrics['loss']:.4f}")
                    print(f"- Perplexity: {eval_metrics['perplexity']:.4f}")
                    
                    # Optionally: track best model
                    if eval_metrics['loss'] < best_val_loss:
                        best_val_loss = eval_metrics['loss']
                        # Save best model
                        model.save_pretrained("best_model")
                    
                    model.train()  # Set back to training mode

                # Log layer statistics
                # if current_step // eval_every > (current_step - 1) // eval_every:
                    avg_layer_losses = torch.where(
                        layer_loss_counts > 0,
                        layer_losses / layer_loss_counts,
                        torch.zeros_like(layer_losses)
                    )
                    
                    enabled_layer = current_step // (total_steps // (2 * len(model.model.layers)))
                    num_enabled = len(model.model.layers) - (len(model.model.layers) - 1 - enabled_layer)
                    
                    print(f"\n=== Step {current_step}/{total_steps} Statistics ===")
                    print(f"Enabled layers: {num_enabled} (from layer {len(model.model.layers)-num_enabled})")
                    print("\nEarly Exit Distribution:")
                    for l in range(len(model.model.layers)):
                        if layer_exit_counts[l] > 0:
                            print(f"Layer {l}: {layer_exit_counts[l]} exits (avg loss: {avg_layer_losses[l]:.4f})")
                    print("=" * 40 + "\n")
                    
                    # Reset statistics
                    layer_losses.zero_()
                    layer_loss_counts.zero_()

            current_step += 1
            
            # After processing batch
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Log metrics every gradient update
            if not is_accumulating:
                avg_batch_time = sum(batch_times[-100:]) / len(batch_times[-100:])
                tokens_per_second = batch_tokens * gradient_accumulation_steps / avg_batch_time
                
                print(f'\nTraining Progress - Epoch: {epoch+1}/{num_epochs}, Step: {current_step}/{total_steps}')
                print(f'Current Loss: {avg_loss:.4f}')
                print(f'Speed Metrics:')
                print(f'- Tokens per second: {tokens_per_second:.2f}')
                print(f'- Average batch time: {avg_batch_time:.3f}s')
                print(f'- Batch size: {train_dataloader.batch_size}')
                print(f'- Sequence length: {batch["input_ids"].shape[1]}')
                
                # Clear metrics periodically
                if len(batch_times) > 1000:
                    batch_times = batch_times[-100:]
        
        # End of epoch metrics
        epoch_time = time.time() - epoch_start_time
        epoch_tokens_per_second = epoch_tokens / epoch_time
        avg_epoch_loss = accumulated_loss / len(train_dataloader)
        
        print(f'\n{"="*20} Epoch {epoch+1}/{num_epochs} completed {"="*20}')
        print(f'Epoch time: {epoch_time:.2f}s')
        print(f'Epoch tokens per second: {epoch_tokens_per_second:.2f}')
        print(f'Average Loss: {avg_epoch_loss:.4f}\n')

    # Final training metrics
    total_training_time = time.time() - training_start_time
    average_tokens_per_second = total_tokens_processed / total_training_time
    
    print(f'\n{"="*20} Training Complete {"="*20}')
    print(f'Total training time: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)')
    print(f'Average tokens per second: {average_tokens_per_second:.2f}')
    print(f'Total tokens processed: {total_tokens_processed:,}')
    print(f'Final Loss: {avg_loss:.4f}\n')
    
    # Return timing metrics for analysis
    return {
        'total_training_time': total_training_time,
        'average_tokens_per_second': average_tokens_per_second,
        'total_tokens_processed': total_tokens_processed,
        'batch_times': batch_times,
        'final_loss': avg_loss
    }

def evaluate(model, val_dataloader, device, num_eval_batches=32):
    """
    Evaluate the model on random subset of validation data
    """
    model.eval()
    total_loss = 0
    
    # Convert dataloader to list and sample random batches
    all_batches = list(val_dataloader)
    eval_batches = random.sample(all_batches, min(num_eval_batches, len(all_batches)))
    
    with torch.no_grad():
        for batch in eval_batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda'):
                outputs = model(**batch)
                loss = outputs.loss
                
            total_loss += loss.item()
    
    avg_loss = total_loss / len(eval_batches)
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity.item()
    }

# Usage example:
"""
# Configure your parameters
config = {
    'gradient_accumulation_steps': 8,
    'micro_batch_size': 8,
    'num_epochs': 3,
    'learning_rate': 1e-4,
    'max_grad_norm': 1.0,

# Initialize your model, optimizer, and scheduler
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

# Initialize gradient scaler
scaler = torch.cuda.amp.GradScaler()

# Train
train_with_gradient_accumulation(
    model=model,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    num_epochs=config['num_epochs'],
    device=device,
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    micro_batch_size=config['micro_batch_size'],
    max_grad_norm=config['max_grad_norm'],
    scheduler=scheduler,
    val_dataloader=val_dataloader,
    eval_every=config['eval_every'],
    scaler=scaler
)
"""