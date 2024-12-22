import time
import torch
from transformers import AutoModelForCausalLM
import random
import gc

def train_baseline(
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
    Train TinyLlama model without modifications, tracking same metrics as layerskip version
    """
    # Start timing for total training
    training_start_time = time.time()
    total_tokens_processed = 0
    batch_times = []
    best_val_loss = float('inf')  # Initialize best_val_loss here
    
    model.train()
    model.gradient_checkpointing_enable()
    total_steps = len(train_dataloader) * num_epochs
    current_step = 0
    accumulated_loss = 0

    # Initialize gradient scaler for mixed precision training
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda')

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        accumulated_loss = 0
        epoch_start_time = time.time()
        epoch_tokens = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            batch_start_time = time.time()
            # More aggressive memory management
            if batch_idx % 50 == 0:  # Changed from 100 to 50
                torch.cuda.empty_cache()
                gc.collect()  # Added explicit garbage collection
            
            # Calculate tokens in batch
            batch_tokens = batch['input_ids'].numel()
            total_tokens_processed += batch_tokens
            epoch_tokens += batch_tokens
            
            # Determine if we should accumulate or update
            is_accumulating = ((batch_idx + 1) % gradient_accumulation_steps != 0) or (batch_idx + 1 == len(train_dataloader))
            
            # Move batch to device and immediately delete original
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

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
                
                optimizer.zero_grad()

                # Free up memory - more aggressive cleanup
                del outputs, loss
                torch.cuda.empty_cache()

                # Calculate average loss
                avg_loss = accumulated_loss / gradient_accumulation_steps
                accumulated_loss = 0

                # Validation
                if (batch_idx + 1) % (gradient_accumulation_steps * eval_every) == 0:
                    print(f'\nRunning validation at step {current_step}...')
                    eval_metrics = evaluate_baseline(model, val_dataloader, device)
                    print(f"Validation metrics:")
                    print(f"- Loss: {eval_metrics['loss']:.4f}")
                    print(f"- Perplexity: {eval_metrics['perplexity']:.4f}")
                    
                    # Save best model and cleanup
                    if eval_metrics['loss'] < best_val_loss:
                        best_val_loss = eval_metrics['loss']
                        model.save_pretrained("best_model")
                    
                    model.train()  # Set back to training mode
                    torch.cuda.empty_cache()  # Added cache cleanup after validation

            current_step += 1
            # Calculate batch metrics
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            # Keep recent batch times only - moved inside loop
            if len(batch_times) > 1000:  # Reduced from 1000 to 100
                batch_times = batch_times[-1000:]
            
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
                
                torch.cuda.empty_cache()  # Added cache cleanup after logging
        
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
    
    return {
        'total_training_time': total_training_time,
        'average_tokens_per_second': average_tokens_per_second,
        'total_tokens_processed': total_tokens_processed,
        'batch_times': batch_times,
        'final_loss': avg_loss
    }


def evaluate_baseline(model, val_dataloader, device, num_eval_batches=32):
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