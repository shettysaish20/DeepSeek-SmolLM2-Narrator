# DeepSeek-SmolLM2-Narrator

A SmolLM2-135M model fine-tuned with DeepSeek architecture modifications (Mixture of Experts and MLHA) to generate Shakespearean-style narration.

## Overview

This project adapts the SmolLM2-135M model, incorporating elements from the DeepSeek architecture like Mixture of Experts (MoE) and Multi-Head Latent Attention (MLHA) to produce text with a distinctive narrative voice reminiscent of Shakespeare. The model is trained using a custom training loop with memory optimizations and expert utilization tracking.

## Contents

*   [config_smollm2_135M.yaml](DeepSeek-SmolLM2-Narrator/config_smollm2_135M.yaml): Configuration file for the model, defining hyperparameters and model architecture.
*   [input.txt](DeepSeek-SmolLM2-Narrator/input.txt): Example input text (Shakespearean dataset) for training the model.
*   [model_deepseek.py](DeepSeek-SmolLM2-Narrator/model_deepseek.py): Modified model definition incorporating DeepSeek architecture, including MoE and custom attention mechanisms.
*   [train.py](DeepSeek-SmolLM2-Narrator/train.py): Training script with mixed precision, gradient accumulation, learning rate scheduling, and expert utilization tracking.
*   [checkpoints/](DeepSeek-SmolLM2-Narrator/checkpoints/): Directory to store model checkpoints during training.

## Key Components

*   **Mixture of Experts (MoE):** Implemented in `model_deepseek.py` to enhance model capacity and performance.  The `DeepSeekMoE` class includes shared and routed experts with dynamic bias updates.
*   **Multi-Head Latent Attention (MLHA):** A custom attention mechanism in `model_deepseek.py` designed for efficient information processing.
*   **Training Loop:** The `train.py` script incorporates memory optimizations such as gradient checkpointing and mixed precision training. It also tracks expert utilization and logs memory usage.

## Training

The `train.py` script trains the model using the Shakespeare dataset. Key features include:

*   Gradient accumulation and mixed precision training for efficient memory usage.
*   Learning rate scheduling with warmup.
*   Expert utilization tracking and dynamic bias updates for MoE layers.
*   Checkpoint saving and loading for resuming training.

To start training:

```bash
python train.py --resume <path_to_checkpoint>