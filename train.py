import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
from pathlib import Path
# from model import LlamaModel
from model_deepseek import LlamaModel
from transformers import AutoTokenizer
import datetime


# Add this function before the train() function
def format_time(seconds):
    """Format seconds into hours, minutes, seconds."""
    return str(datetime.timedelta(seconds=int(seconds)))


# Load config
with open('config_smollm2_135M.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Constants from config
HIDDEN_SIZE = config['model']['model_config']['hidden_size']
NUM_HEADS = config['model']['model_config']['num_attention_heads']
NUM_LAYERS = config['model']['model_config']['num_hidden_layers']
SEQ_LEN = config['tokens']['sequence_length']
VOCAB_SIZE = config['model']['model_config']['vocab_size']
MICRO_BATCH_SIZE = 4  # Reduced from 16 for 8GB VRAM
GRAD_ACCUMULATION_STEPS = 4  # Accumulate 4 steps to simulate batch size 16
LEARNING_RATE = config['optimizer']['learning_rate_scheduler']['learning_rate']
WEIGHT_DECAY = config['optimizer']['weight_decay']
WARMUP_STEPS = 1000  # Shorter warmup for quicker training
TOTAL_STEPS = 5000  # As requested

deepseek_config = {
    "compression_ratio": 4,
    "num_experts": 4,
    "num_shared_experts": 1,
    "top_k": 2
}

# Shakespeare dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Load text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize entire text
        self.tokenized_text = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids[0]
        
        # Ensure we have enough data
        self.num_samples = max(len(self.tokenized_text) // self.seq_len - 1, 1)
        
    def __len__(self):
        return self.num_samples * self.seq_len
    
    def __getitem__(self, idx):
        # Get starting position
        start_idx = idx % (len(self.tokenized_text) - self.seq_len - 1)
        
        # Get input and target
        input_seq = self.tokenized_text[start_idx:start_idx + self.seq_len]
        target_seq = self.tokenized_text[start_idx + 1:start_idx + self.seq_len + 1]
        
        # Pad if necessary
        if len(input_seq) < self.seq_len:
            pad_len = self.seq_len - len(input_seq)
            input_seq = torch.cat([input_seq, torch.zeros(pad_len, dtype=torch.long)])
            target_seq = torch.cat([target_seq, torch.zeros(pad_len, dtype=torch.long)])
            
        return {"input_ids": input_seq, "labels": target_seq}

def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup and then linear decay learning rate scheduler."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0 #max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)


# Add this function before the train() function
@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, device="cuda"):
    """Generate text from a prompt."""
    model.eval()
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    
    # Initialize generation
    generated_ids = input_ids[0].tolist()
    
    # Generate text
    for _ in range(max_length):
        # Get the last SEQ_LEN tokens (or all if less than SEQ_LEN)
        curr_input_ids = torch.tensor([generated_ids[-min(SEQ_LEN, len(generated_ids)):]], dtype=torch.long).to(device)
        
        # Forward pass
        with autocast(device_type="cuda"):
            logits, _ = model(curr_input_ids)
        
        # Get the logits for the last token
        next_token_logits = logits[-1, :] / temperature
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        
        # Sample the next token
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        # Add the token to the generated sequence
        generated_ids.append(next_token)
        
        # Stop if we generate an EOS token
        if next_token == tokenizer.eos_token_id:
            break
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    model.train()
    return generated_text

def calculate_model_params(model):
    """Calculate and print model parameter counts by category."""
    total_params = 0
    attn_params = 0
    mlp_params = 0
    moe_params = 0
    embedding_params = 0
    norm_params = 0
    other_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if 'self_attn' in name:
            attn_params += param_count
        elif 'mlp' in name:
            if 'moe' in name:
                moe_params += param_count
            else:
                mlp_params += param_count
        elif 'embed' in name:
            embedding_params += param_count
        elif 'norm' in name:
            norm_params += param_count
        else:
            other_params += param_count
    
    print(f"\n{'='*50}")
    print(f"Model Parameter Breakdown:")
    print(f"{'='*50}")
    print(f"Total Parameters: {total_params/1e6:.2f}M")
    print(f"  - Attention Params: {attn_params/1e6:.2f}M ({attn_params/total_params*100:.1f}%)")
    print(f"  - MLP Params: {mlp_params/1e6:.2f}M ({mlp_params/total_params*100:.1f}%)")
    print(f"  - MoE Params: {moe_params/1e6:.2f}M ({moe_params/total_params*100:.1f}%)")
    print(f"  - Embedding Params: {embedding_params/1e6:.2f}M ({embedding_params/total_params*100:.1f}%)")
    print(f"  - Normalization Params: {norm_params/1e6:.2f}M ({norm_params/total_params*100:.1f}%)")
    print(f"  - Other Params: {other_params/1e6:.2f}M ({other_params/total_params*100:.1f}%)")
    print(f"{'='*50}\n")
    
    return total_params/1e6  # Return total in millions

# Inside the train function, after model initialization but before moving to device:
def apply_memory_optimizations(model):
    """Apply additional memory optimizations for the DeepSeek model."""
    # Increase gradient checkpointing coverage
    for layer in model.layers:
        # Enable gradient checkpointing for MoE layers
        if hasattr(layer.mlp, 'moe'):
            for expert in layer.mlp.moe.routed_experts:
                setattr(expert, 'gradient_checkpointing', True)
            for expert in layer.mlp.moe.shared_experts:
                setattr(expert, 'gradient_checkpointing', True)
    
    # Set a smaller batch size and increase gradient accumulation
    # global MICRO_BATCH_SIZE, GRAD_ACCUMULATION_STEPS
    # MICRO_BATCH_SIZE = 2  # Reduce batch size from 4 to 2
    # GRAD_ACCUMULATION_STEPS = 8  # Increase accumulation to maintain effective batch size

    # print(f"Memory optimizations applied: Batch size = {MICRO_BATCH_SIZE}, Grad accumulation = {GRAD_ACCUMULATION_STEPS}")


# 6. Modify the training loop to track expert utilization for MoE
def track_expert_utilization(model, writer, global_step):
    """Track expert utilization metrics for MoE."""
    if not hasattr(model, 'layers'):
        return
    
    # Initialize counters
    expert_counts = torch.zeros(4)  # For 8 experts
    total_tokens = 0
    
    # Sample a batch and track which experts are selected
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer.mlp, 'moe'):
            moe = layer.mlp.moe
            # Get routing probabilities for a sample batch
            with torch.no_grad():
                # Create a small random batch for testing
                sample_batch = torch.randint(0, 100, (MICRO_BATCH_SIZE, 32)).to(next(model.parameters()).device)
                sample_embeds = model.embed_tokens(sample_batch)
                # Forward through layers up to current one
                x = sample_embeds
                for i in range(layer_idx):
                    x = model.layers[i](x)
                # Get input to MoE
                x = layer.input_layernorm(x)
                x = layer.self_attn(x)
                x = x + sample_embeds
                x = layer.post_attention_layernorm(x)
                
                # Get routing decisions
                routing_logits = moe.router(x) + moe.routing_bias
                routing_probs = torch.sigmoid(routing_logits)
                _, indices = torch.topk(routing_probs, moe.top_k, dim=-1)
                
                # Count expert selections
                for k in range(moe.top_k):
                    expert_indices = indices[..., k].flatten()
                    for i in range(moe.num_routed_experts):
                        expert_counts[i] += (expert_indices == i).sum().item()
                
                total_tokens += MICRO_BATCH_SIZE * 32
    
    if total_tokens > 0:
        # Normalize and log
        expert_probs = expert_counts / (total_tokens * moe.top_k)
        for i in range(len(expert_probs)):
            writer.add_scalar(f'moe/expert_{i}_utilization', expert_probs[i], global_step)
        
        # Calculate load balancing metrics
        expert_entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-10))
        writer.add_scalar('moe/expert_entropy', expert_entropy, global_step)

# 7. Add memory usage tracking in the main training loop
# Inside the training loop, add:
def log_memory_usage(writer, global_step):
    """Log detailed GPU memory usage statistics."""
    if torch.cuda.is_available():
        # Current allocation
        allocated = torch.cuda.memory_allocated() / 1024**2
        # Maximum allocation
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        # Cache allocation
        reserved = torch.cuda.memory_reserved() / 1024**2
        # Maximum cache allocation
        max_reserved = torch.cuda.max_memory_reserved() / 1024**2
        
        writer.add_scalar('memory/allocated_mb', allocated, global_step)
        writer.add_scalar('memory/max_allocated_mb', max_allocated, global_step)
        writer.add_scalar('memory/reserved_mb', reserved, global_step)
        writer.add_scalar('memory/max_reserved_mb', max_reserved, global_step)
        
        # Log available memory
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
        available = total_memory - reserved
        writer.add_scalar('memory/available_mb', available, global_step)
        
        return {
            "allocated": allocated,
            "max_allocated": max_allocated,
            "reserved": reserved,
            "max_reserved": max_reserved,
            "total": total_memory,
            "available": available
        }
    return {}


def train(resume_from = None):
    torch.backends.cudnn.benchmark = True
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable TF32 precision (faster on Ampere+ GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set up memory profiling
    starting_gpu_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"Starting GPU memory: {starting_gpu_memory:.2f} MB")
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir='runs/smollm2_shakespeare')
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['tokenizer_name_or_path'])
    
    # Initialize model with memory optimizations
    # model = LlamaModel(config['model'])
    model = LlamaModel(config['model'], deepseek_config)

    model_params_m = calculate_model_params(model)
    apply_memory_optimizations(model)
    
    # Apply gradient checkpointing to save memory
    def enable_gradient_checkpointing(model):
        # Enable for decoder layers
        for layer in model.layers:
            layer.gradient_checkpointing = True
            
    # Define custom forward method with gradient checkpointing
    def custom_forward(self, x):
        if hasattr(self, 'gradient_checkpointing') and self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            residual = x
            x = self.input_layernorm(x)
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.self_attn), x)
            x = x + residual
            residual = x
            x = self.post_attention_layernorm(x)
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mlp), x)
            x = x + residual
            return x
        else:
            return self._original_forward(x)
    
    # Save original forward
    for layer in model.layers:
        layer._original_forward = layer.forward
        layer.forward = custom_forward.__get__(layer, layer.__class__)
    
    enable_gradient_checkpointing(model)
    
    # Move model to device
    model = model.to(device)
    
    # Print model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count/1e6:.2f}M")
    
    # Set up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(config['optimizer']['optimizer_factory']['adam_beta1'],
               config['optimizer']['optimizer_factory']['adam_beta2']),
        eps=config['optimizer']['optimizer_factory']['adam_eps'],
    )
    
    # Set up scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, TOTAL_STEPS)
    
    # Set up mixed precision training
    scaler = GradScaler()

    # Add this after initializing model, optimizer, scheduler, and scaler
    global_step = 0
    epoch = 0
    best_loss = float('inf')
    
    # Load dataset
    shakespeare_file = "input.txt"  # Update with your Shakespeare file path
    dataset = ShakespeareDataset(shakespeare_file, tokenizer, SEQ_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        num_workers=8,  # Adjust based on your CPU
        pin_memory=True
    )
    
    # Training loop
    global_step = 0
    epoch = 0
    best_loss = float('inf')
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Set up profiler
    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(f"Profiler results:\n{output}")
        p.export_chrome_trace(f"trace_{global_step}.json")
    
    ## Initialize start_time and steps_per_second
    start_time = time.time()
    steps_per_second = 0

    # Load checkpoint if resuming    
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler'])
        
        global_step = checkpoint['global_step']
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        
        print(f"Resumed from step {global_step}")
        
        # Adjust start time based on progress
        start_time = time.time() - (global_step / 3.0)  # Rough estimate of time elapsed
    
    # Main training loop
    print("Starting training...")
    prev_total = start_time
    while global_step < TOTAL_STEPS:
        epoch_start_time = time.time()
        epoch += 1
        print(f"Starting epoch {epoch}")
        
        total_loss = 0.0
        samples_seen = 0
        # Iterate through batches
        for batch_idx, batch in enumerate(dataloader):
            # Profile every 500 steps
            use_profiler = global_step % 500 == 0 and global_step > 0
            
            profile_ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=trace_handler,
                record_shapes=True,
                with_stack=True,
            ) if use_profiler else torch.cuda.profiler.profile()
            
            with profile_ctx:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass with mixed precision
                with autocast(device_type="cuda"):
                    _, loss = model(input_ids, labels)
                    loss = loss / GRAD_ACCUMULATION_STEPS  # Normalize for gradient accumulation
                
                # Add NaN check before backward pass
                if torch.isnan(loss):
                    print(f"Warning: NaN loss encountered at global step {global_step}. Skipping update.")
                    optimizer.zero_grad()
                    continue
                
                # Backward pass with mixed precision
                scaler.scale(loss).backward()
                
                # Step optimizer every GRAD_ACCUMULATION_STEPS batches
                if (batch_idx + 1) % GRAD_ACCUMULATION_STEPS == 0 or batch_idx == len(dataloader) - 1:
                    print(global_step)
                    # Gradient clipping
                    if config['optimizer']['clip_grad'] > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad'])
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    # Increment global step
                    global_step += 1
                    # Print stats
                    if global_step % 10 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        gpu_memory = torch.cuda.memory_allocated() / 1024**2
                        
                        # Calculate timing information
                        elapsed = time.time() - prev_total
                        prev_total+=elapsed
                        total_elapsed = time.time() - start_time

                        steps_per_second = global_step / max(1, total_elapsed)
                        eta_seconds = (TOTAL_STEPS - global_step) / max(0.1, steps_per_second)
                        
                        print(f"Step {global_step}/{TOTAL_STEPS} | Update time: {format_time(elapsed)} | Elapsed Since Start: {format_time(total_elapsed)} | Loss: {loss.item()*GRAD_ACCUMULATION_STEPS:.4f} | "
                            f"LR: {current_lr:.6f} | GPU: {gpu_memory:.2f}MB | "
                            f"Speed: {steps_per_second:.2f} steps/s | ETA: {format_time(eta_seconds)}")
                        
                        # Log to tensorboard
                        writer.add_scalar('train/loss', loss.item()*GRAD_ACCUMULATION_STEPS, global_step)
                        writer.add_scalar('train/lr', current_lr, global_step)
                        writer.add_scalar('system/gpu_memory_mb', gpu_memory, global_step)
                    
                    
                    # Add periodic MoE tracking inside the training loop
                    if global_step % 100 == 0:
                        track_expert_utilization(model, writer, global_step)
                        mem_stats = log_memory_usage(writer, global_step)
                        print(f"Memory Usage: {mem_stats['allocated']:.1f}MB allocated, "
                            f"{mem_stats['available']:.1f}MB available")

                    # Save checkpoint
                    if global_step % 100 == 0:
                        checkpoint_path = f"checkpoints/smollm2_shakespeare_step{global_step}.pt"
                        torch.save({
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'loss': loss.item(),
                        }, checkpoint_path)
                        print(f"Checkpoint saved: {checkpoint_path}")
                    
                    if global_step % 50 == 0:
                        # Generate sample text
                        test_prompt = "Before we proceed any further, "
                        generated_text = generate_text(model, tokenizer, test_prompt)
                        
                        print("\n" + "="*50)
                        print(f"Sample generation at step {global_step}:")
                        print(f"Prompt: {test_prompt}")
                        print(f"Generated: {generated_text}")
                        print("="*50 + "\n")
                        
                        # Log to tensorboard
                        writer.add_text('generation', generated_text, global_step)
                    
                    # Replace the MoE routing bias update block with the following:
                    if global_step % 50 == 0:
                        with torch.no_grad():
                            # Recompute embeddings from the current batch
                            x_sample = model.embed_tokens(input_ids)
                            # Pass through each decoder layer to update routing bias in MoE layers
                            for layer in model.layers:
                                if hasattr(layer.mlp, 'moe'):
                                    moe = layer.mlp.moe
                                    x_in = layer.input_layernorm(x_sample)
                                    x_in = layer.self_attn(x_in)
                                    routing_logits = moe.router(x_in) + moe.routing_bias
                                    routing_probs = torch.sigmoid(routing_logits)
                                    expert_load = routing_probs.mean(dim=[0, 1])
                                    moe.update_bias_terms(expert_load)
                                x_sample = layer(x_sample)
                    
                    # Check if we've reached total steps
                    if global_step >= TOTAL_STEPS:
                        break
                
                # Update running loss
                total_loss += loss.item() * GRAD_ACCUMULATION_STEPS
                samples_seen += input_ids.size(0)
        
        # End of epoch stats
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / (batch_idx + 1)
        print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint at end of epoch
        checkpoint_path = f"checkpoints/smollm2_shakespeare_epoch{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Epoch checkpoint saved: {checkpoint_path}")
    
    # Save final model
    torch.save(model.state_dict(), "checkpoints/smollm2_shakespeare_final.pt")
    print("Training completed!")
    
    # Final stats
    final_gpu_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"Final GPU memory: {final_gpu_memory:.2f} MB")
    print(f"Memory increase during training: {final_gpu_memory - starting_gpu_memory:.2f} MB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SmolLM2 on Shakespeare')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    train(resume_from=args.resume)
