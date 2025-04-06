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
MICRO_BATCH_SIZE = 16  # Adjusted back to original as we have more GPU memory
GRAD_ACCUMULATION_STEPS = 1  # Adjusted back to original
LEARNING_RATE = config['optimizer']['learning_rate_scheduler']['learning_rate']
WEIGHT_DECAY = config['optimizer']['weight_decay']
WARMUP_STEPS = config['optimizer']['learning_rate_scheduler']['lr_warmup_steps']
TOTAL_STEPS = config['tokens']['train_steps']

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

        return {"input_ids": input_seq, "labels": target_seq}

def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup and then linear decay learning rate scheduler."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

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
        next_token_logits = logits[:, -1, :] / temperature

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

    # Enable gradient checkpointing for all layers
    for layer in model.layers:
        layer.gradient_checkpointing = True

    # Set a smaller batch size and increase gradient accumulation
    # global MICRO_BATCH_SIZE, GRAD_ACCUMULATION_STEPS
    # MICRO_BATCH_SIZE = 2  # Reduce batch size from 4 to 2
    # GRAD_ACCUMULATION_STEPS = 8  # Increase accumulation to maintain effective batch size

    # print(f"Memory optimizations applied: Batch size = {MICRO_BATCH_SIZE}, Grad accumulation = {GRAD_ACCUMULATION_STEPS}")


# 6. Modify the training loop to track expert utilization for MoE
def track_expert_utilization(model, writer, global_step, rank):
    """Track expert utilization metrics for MoE."""
    if rank != 0 or not hasattr(model, 'layers'):
        return

    # Initialize counters
    expert_counts = torch.zeros(4, device=f'cuda:{rank}')  # For 4 experts
    total_tokens = 0

    # Sample a batch and track which experts are selected
    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer.mlp, 'moe'):
            moe = layer.mlp.moe
            # Get routing probabilities for a sample batch
            with torch.no_grad():
                # Create a small random batch for testing
                sample_batch = torch.randint(0, 100, (MICRO_BATCH_SIZE, 32)).to(f'cuda:{rank}')
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
                        expert_counts[i] += (expert_indices == i).sum()

                total_tokens += MICRO_BATCH_SIZE * 32

    if total_tokens > 0:
        # Normalize and log
        expert_probs = expert_counts.float() / (total_tokens * moe.top_k)
        for i in range(len(expert_probs)):
            writer.add_scalar(f'moe/expert_{i}_utilization', expert_probs[i], global_step)

        # Calculate load balancing metrics
        expert_entropy = -torch.sum(expert_probs * torch.log(expert_probs + 1e-10))
        writer.add_scalar('moe/expert_entropy', expert_entropy, global_step)

# 7. Add memory usage tracking in the main training loop
# Inside the training loop, add:
def log_memory_usage(writer, global_step, rank):
    """Log detailed GPU memory usage statistics."""
    if rank == 0 and torch.cuda.is_available():
        # Current allocation
        allocated = torch.cuda.memory_allocated(rank) / 1024**2
        # Maximum allocation
        max_allocated = torch.cuda.max_memory_allocated(rank) / 1024**2
        # Cache allocation
        reserved = torch.cuda.memory_reserved(rank) / 1024**2
        # Maximum cache allocation
        max_reserved = torch.cuda.max_memory_reserved(rank) / 1024**2

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


def train(rank, world_size, resume_from = None):
    torch.backends.cudnn.benchmark = True
    # Set device
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}/{world_size} using device: {device}")

    # Initialize distributed process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    # Enable TF32 precision (faster on Ampere+ GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set up memory profiling
    starting_gpu_memory = torch.cuda.memory_allocated(rank) / 1024**2
    print(f"Rank {rank}: Starting GPU memory: {starting_gpu_memory:.2f} MB")

    # Set up tensorboard (only on rank 0)
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir='runs/smollm2_shakespeare')

    # Initialize tokenizer (only need one instance)
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['tokenizer_name_or_path'])

    # Initialize model with memory optimizations
    model = LlamaModel(config['model'], deepseek_config)

    model_params_m = calculate_model_params(model)
    apply_memory_optimizations(model)

    # Move model to device
    model = model.to(device)
    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank])

    # Print model size (only on rank 0)
    if rank == 0:
        param_count = sum(p.numel() for p in model.module.parameters())
        print(f"Model parameters: {param_count/1e6:.2f}M")

    # Set up optimizer
    optimizer = AdamW(
        model.module.parameters(),
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
    # Create DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,  # Shuffle is handled by DistributedSampler
        num_workers=8,  # Adjust based on your CPU
        pin_memory=True,
        sampler=sampler
    )

    # Create checkpoints directory (only on rank 0)
    if rank == 0:
        os.makedirs('checkpoints', exist_ok=True)

    # Set up profiler (only on rank 0)
    def trace_handler(p):
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(f"Rank {rank}: Profiler results:\n{output}")
        p.export_chrome_trace(f"trace_rank_{rank}_step_{global_step}.json")

    ## Initialize start_time and steps_per_second (only on rank 0)
    start_time = time.time() if rank == 0 else 0
    steps_per_second = 0

    # Load checkpoint if resuming
    if resume_from is not None and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler'])
        global_step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        print(f"Rank {rank}: Resumed from step {global_step}")
        if rank == 0:
            start_time = time.time() - (global_step * (time.time() - start_time) / max(1, global_step)) # Adjust start time

    # Main training loop
    print(f"Rank {rank}: Starting training...")
    prev_total = time.time()
    while global_step < TOTAL_STEPS:
        epoch_start_time = time.time()
        epoch += 1
        if rank == 0:
            print(f"Epoch {epoch}")
        sampler.set_epoch(epoch) # Important for DistributedSampler

        total_loss = 0.0
        samples_seen = 0
        # Iterate through batches
        for batch_idx, batch in enumerate(dataloader):
            # Profile every 500 steps (only on rank 0)
            use_profiler = rank == 0 and global_step % 500 == 0 and global_step > 0

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
                    print(f"Rank {rank}: Warning: NaN loss encountered at global step {global_step}. Skipping update.")
                    optimizer.zero_grad()
                    continue

                # Backward pass with mixed precision
                scaler.scale(loss).backward()

                # Step optimizer every GRAD_ACCUMULATION_STEPS batches
                if (batch_idx + 1) % GRAD_ACCUMULATION_STEPS == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['clip_grad'])

                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    # Increment global step
                    global_step += 1
                    # Print stats (only on rank 0)
                    if rank == 0 and global_step % 10 == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        gpu_memory = torch.cuda.memory_allocated(rank) / 1024**2

                        # Calculate timing information
                        elapsed = time.time() - prev_total
                        prev_total += elapsed
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

                        # Add periodic MoE tracking
                        if global_step % 100 == 0:
                            track_expert_utilization(model.module, writer, global_step, rank)
                            mem_stats = log_memory_usage(writer, global_step, rank)
                            if mem_stats:
                                print(f"Rank {rank}: Memory Usage: {mem_stats['allocated']:.1f}MB allocated, "
                                      f"{mem_stats['available']:.1f}MB available")

                        # Save checkpoint
                        if global_step % 100 == 0:
                            checkpoint_path = f"checkpoints/smollm2_shakespeare_step{global_step}.pt"
                            torch.save({
                                'global_step': global_step,
                                'epoch': epoch,
                                'model_state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'scaler': scaler.state_dict(),
                                'loss': loss.item(),
                            }, checkpoint_path)
                            print(f"Rank 0: Checkpoint saved: {checkpoint_path}")

                        if global_step % 50 == 0:
                            # Generate sample text
                            test_prompt = "Before we proceed any further, "
                            generated_text = generate_text(model.module, tokenizer, test_prompt, device=device)

                            print("\n" + "="*50)
                            print(f"Rank 0: Sample generation at step {global_step}:")
                            print(f"Prompt: {test_prompt}")
                            print(f"Generated: {generated_text}")
                            print("="*50 + "\n")

                            # Log to tensorboard
                            writer.add_text('generation', generated_text, global_step)

                    # Check if we've reached total steps
                    if global_step >= TOTAL_STEPS:
                        break
            if global_step >= TOTAL_STEPS:
                break

        # End of epoch stats (only on rank 0)
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / (batch_idx + 1) if (batch_idx + 1) > 0 else 0
        if rank == 0:
            print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds | Avg Loss: {avg_loss:.4f}")

            # Save checkpoint at end of epoch
            checkpoint_path = f"checkpoints/smollm2_shakespeare_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Rank 0: Epoch checkpoint saved: {checkpoint_path}")

        if global_step >= TOTAL_STEPS:
            break

    # Save final model (only on rank 0)
    if rank == 0:
        torch.save(model.module.state_dict(), "checkpoints/smollm2_shakespeare_final.pt")
        print("Training completed!")

        # Final stats
        final_gpu_memory = torch.cuda.memory_allocated(rank) / 1024**2
        print(f"Final GPU memory (Rank 0): {final_gpu_memory:.2f} MB")
        print(f"Memory increase during training (Rank 0): {final_gpu_memory - starting_gpu_memory:.2f} MB")

    dist.destroy_process_group()

def main(resume_from):
    world_size = torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size, resume_from,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train SmolLM2 on Shakespeare with DDP')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    main(args.resume)