# DeepSeek-SmolLM2-Narrator

A SmolLM2-135M model fine-tuned with DeepSeek architecture modifications (Mixture of Experts and MLHA) to generate Shakespearean-style narration.

## Overview

This project adapts the SmolLM2-135M model, incorporating elements from the DeepSeek architecture like Mixture of Experts (MoE) and Multi-Head Latent Attention (MLHA) to produce text with a distinctive narrative voice reminiscent of Shakespeare. The model is trained using a custom training loop with memory optimizations and expert utilization tracking.

## Contents

*   [config_smollm2_135M.yaml](DeepSeek-SmolLM2-Narrator/config_smollm2_135M.yaml): Configuration file for the model, defining hyperparameters and model architecture.
*   [input.txt](DeepSeek-SmolLM2-Narrator/input.txt): Example input text (Shakespearean dataset) for training the model.
*   [model_deepseek.py](DeepSeek-SmolLM2-Narrator/model_deepseek.py): Modified model definition incorporating DeepSeek architecture, including MoE and custom attention mechanisms.
*   [train.py](DeepSeek-SmolLM2-Narrator/train.py): Training script with mixed precision, gradient accumulation, learning rate scheduling, and expert utilization tracking.
*   [train_ddp.py](DeepSeek-SmolLM2-Narrator/train_ddp.py): Training script along with Distributed Data Parallel Training to enable faster training time in multiple GPUs
*   [checkpoints/](DeepSeek-SmolLM2-Narrator/checkpoints/): Directory to store model checkpoints during training.

## Model Configuration

```bash
==================================================
Model Parameter Breakdown:
==================================================
Total Parameters: 373.03M
  - Attention Params: 26.13M (7.0%)
  - MLP Params: 0.00M (0.0%)
  - MoE Params: 318.56M (85.4%)
  - Embedding Params: 28.31M (7.6%)
  - Normalization Params: 0.04M (0.0%)
  - Other Params: 0.00M (0.0%)
==================================================

```

## Key Components

*   **Mixture of Experts (MoE):** Implemented in `model_deepseek.py` to enhance model capacity and performance.  The `DeepSeekMoE` class includes shared and routed experts with dynamic bias updates.
*   **Multi-Head Latent Attention (MLHA):** A custom attention mechanism in `model_deepseek.py` designed for efficient information processing.
*   **Training Loop:** The `train.py` script incorporates memory optimizations such as gradient checkpointing and mixed precision training. It also tracks expert utilization and logs memory usage.

## Training Steps

1. ### Local Training
    - From Global Step 0 to Global Step 700
        - GPU: NVIDIA RTX 4060 
        - Gradient Accumulation steps: 4
        - Total number of Steps: 700 * 4 = 2800
        - Average time taken: 5 min / 10 steps
2. ### AWS EC2 Training
    - From Global Step 700 to Global Step 1800 (resumed checkpoint)
        - Instance type: g4dn.12xlarge
        - Gradient Accumulation steps: 8
        - Total number of Steps: 1100 * 8 = 8800
        - Average time taken: 1 min / 10 steps (**80%** reduction compared to Local Training Speed)

Total number of steps trained: 2800 + 8800 = **11600**

## Training Challenges
1. ### NaN loss error
    - Had to stop further training because of the NaN loss Error.
        ![AWS NaN Error](training_logs_ec2/NaN%20error.png)
    - Fixes tried:
        - Reducing **Learning Rate** and **Gradient Clipping Max Norm**
        - Did in-depth debugging of the DeepSeek model layers to check for **exploding gradients**
        - Shifted training from a single-GPU system to a **multi-GPU system** to ensure higher precision calculations
    - Possible Reasons for the Issue:
        - Using Shakespeare dataset (with 100k lines) might not be enough for a model of this size (373m paremeters)
            - Trying re-training with a larger text dataset such as Cosmopedia V2


## Training methodology

Both the `train.py` and `train_ddp.py` script trains the model using the Shakespeare dataset. Key features include:

*   Gradient accumulation and mixed precision training for efficient memory usage.
*   Learning rate scheduling with warmup.
*   Expert utilization tracking and dynamic bias updates for MoE layers.
*   Checkpoint saving and loading for resuming training.
*   For distributed data parallel training, use `train_ddp.py` to enable faster training time in multiple GPUs.

To start training:

```bash
python train.py --resume <path_to_checkpoint>
```
```bash
python train_ddp.py --resume <path_to_checkpoint>
```

## Training logs + Sample Output Generation

*Local Training*
```bash
Step 460/5000 | Update time: 0:05:08 | Elapsed Since Start: 3:56:08 | Loss: 5.1720 | LR: 0.002780 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:36:40
Step 470/5000 | Update time: 0:05:11 | Elapsed Since Start: 4:01:20 | Loss: 4.5980 | LR: 0.002773 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:35:00
Step 480/5000 | Update time: 0:05:10 | Elapsed Since Start: 4:06:30 | Loss: 4.5400 | LR: 0.002767 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:33:20
Step 490/5000 | Update time: 0:05:15 | Elapsed Since Start: 4:11:45 | Loss: 4.6133 | LR: 0.002761 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:31:40
Step 500/5000 | Update time: 0:05:19 | Elapsed Since Start: 4:17:05 | Loss: 4.9827 | LR: 0.002755 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:30:00
Memory Usage: 4512.8MB allocated, -1234.5MB available
Checkpoint saved: checkpoints/smollm2_shakespeare_step500.pt

==================================================
Sample generation at step 500:
Prompt: Before we proceed any further,
Generated: Before we proceed any further,





,
then; nurse
 ruin,
To lord,

 on elset
 home,

 away our father swear of thingstrous foreoeb, I will thee
I, I to do roundua together;
Being heaven the visit the kingfeit and n
May they walk in a Juliet noble thingrows
Till beicial death,
And seem up, the help, times to I, of all,
circ bark was thay
==================================================

Step 510/5000 | Update time: 0:05:32 | Elapsed Since Start: 4:22:37 | Loss: 4.9262 | LR: 0.002749 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:28:20
Step 520/5000 | Update time: 0:05:16 | Elapsed Since Start: 4:27:54 | Loss: 5.2023 | LR: 0.002743 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:26:40
Step 530/5000 | Update time: 0:05:15 | Elapsed Since Start: 4:33:09 | Loss: 5.5463 | LR: 0.002737 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:25:00
Step 540/5000 | Update time: 0:05:05 | Elapsed Since Start: 4:38:14 | Loss: 5.6505 | LR: 0.002731 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:23:20
Step 550/5000 | Update time: 0:05:07 | Elapsed Since Start: 4:43:22 | Loss: 5.9054 | LR: 0.002724 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:21:40

==================================================
Sample generation at step 550:
Prompt: Before we proceed any further,
Generated: Before we proceed any further,  confessonrel myself on
O what,

RIcy:

D'd

 my,

G cour not

CUREL usurru is was with I.

KATHAN:
 we I have thee him,
 is that they me no I me. his

 aT,

 no, about I will bear off the right. you
I prayers will possession:

KoyLO:
But,LO
 purge, the
==================================================

Step 560/5000 | Update time: 0:05:17 | Elapsed Since Start: 4:48:39 | Loss: 5.6426 | LR: 0.002718 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:20:00
Step 570/5000 | Update time: 0:04:44 | Elapsed Since Start: 4:53:24 | Loss: 5.6972 | LR: 0.002712 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:18:20
Step 580/5000 | Update time: 0:05:04 | Elapsed Since Start: 4:58:28 | Loss: 5.5846 | LR: 0.002706 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:16:40
Step 590/5000 | Update time: 0:05:07 | Elapsed Since Start: 5:03:36 | Loss: 5.7719 | LR: 0.002700 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:15:00
Step 600/5000 | Update time: 0:05:07 | Elapsed Since Start: 5:08:43 | Loss: 6.2422 | LR: 0.002694 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:13:20
Memory Usage: 4512.8MB allocated, -1234.5MB available
Checkpoint saved: checkpoints/smollm2_shakespeare_step600.pt

==================================================
Sample generation at step 600:
Prompt: Before we proceed any further,
Generated: Before we proceed any further, , one, my love. as t themGDogsUUC?:,US winter for haveou, cares look are serve, fair inJ she;
OL where havest, must she,
To wandering, they aian-- to nighting you left must king mine lie



T feasts; him I

 thereThat fair me so,
 be..: the to crave was

: p!.
US:
 it a one to good:
==================================================

Step 610/5000 | Update time: 0:05:34 | Elapsed Since Start: 5:14:18 | Loss: 5.8052 | LR: 0.002688 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:11:40
Step 620/5000 | Update time: 0:05:07 | Elapsed Since Start: 5:19:25 | Loss: 5.1442 | LR: 0.002682 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:10:00
Step 630/5000 | Update time: 0:04:52 | Elapsed Since Start: 5:24:17 | Loss: 5.4212 | LR: 0.002676 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:08:20
Step 640/5000 | Update time: 0:05:03 | Elapsed Since Start: 5:29:21 | Loss: 5.7623 | LR: 0.002669 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:06:40
Step 650/5000 | Update time: 0:04:59 | Elapsed Since Start: 5:34:21 | Loss: 5.4537 | LR: 0.002663 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:05:00

==================================================
Sample generation at step 650:
Prompt: Before we proceed any further,
Generated: Before we proceed any further, .

DUL ifY:
Well see and name'd been be'd,
A say our never the senate, if,
Unless is her they if'd give truebal
Which a, b not other thiseringday
With not a the me of won comeatter;

KING LEARI nightThIONUMRYOL, to his ifIO,
elf good.

PoorUCESTLO:
I'll her children.

LENUT B
==================================================

Step 660/5000 | Update time: 0:05:18 | Elapsed Since Start: 5:39:39 | Loss: 5.5931 | LR: 0.002657 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:03:20
Step 670/5000 | Update time: 0:05:04 | Elapsed Since Start: 5:44:43 | Loss: 5.1702 | LR: 0.002651 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:01:40
Step 680/5000 | Update time: 0:04:58 | Elapsed Since Start: 5:49:42 | Loss: 5.2709 | LR: 0.002645 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 12:00:00
Step 690/5000 | Update time: 0:05:06 | Elapsed Since Start: 5:54:48 | Loss: 5.2998 | LR: 0.002639 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 11:58:20
Step 700/5000 | Update time: 0:05:02 | Elapsed Since Start: 5:59:51 | Loss: 5.2226 | LR: 0.002633 | GPU: 4512.84MB | Speed: 0.03 steps/s | ETA: 11:56:40
Memory Usage: 4512.8MB allocated, -1234.5MB available
Checkpoint saved: checkpoints/smollm2_shakespeare_step700.pt

==================================================
Sample generation at step 700:
Prompt: Before we proceed any further,
Generated: Before we proceed any further,
To the be blood,
What thou time his one king.
I love you not to people to the suit d times
Should a o-,
My all not will not my bos which son as
To all for such and ' from.

YIDI God A worldly out
Yet live is hear I so by his your gone.

GLOAN dead come,?:
I thenuck the f Y:
Alardon were found; you their ease
==================================================
```

*AWS EC2 Training*

![AWS EC2 Training](training_logs_ec2/Screenshot%202025-04-06%20211830.png)