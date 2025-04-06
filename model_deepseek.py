######################################################################
# The School of AI- Session 15 Assignment
# This is the re-implementation of the SmolLM2-135M model modified
# with DeepSeek architecture
######################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import SiLU
import math
import yaml

def _init_weights(module, std=0.041666666666666664):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=std)
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)

class RotaryPositionalEmbedding(nn.Module):
    """
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L240
    Rotary Positional Embedding (RoPE) for transformers Implemntation derived from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    """
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def compute_rotary_embeddings(self, seq_len: int, device):
        dim = self.dim // 2  # use half dimension
        positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, dtype=torch.float32, device=device) / dim))
        sinusoid_inp = torch.matmul(positions, inv_freq.unsqueeze(0))
        sin = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(2)  # shape (1, seq_len, 1, dim)
        cos = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(2)
        return sin, cos

    def apply_rotary_emb(self, x, rotary_embeddings):
        sin, cos = rotary_embeddings
        # x is expected to be of shape [B, T, num_heads, D] where D should be even.
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated
    

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, rotary_emb, config, compression_ratio=4): # hidden_size, num_heads,
        super().__init__()
        self.config = config
        self.hidden_size = self.config["hidden_size"] #hidden_size
        self.num_heads = self.config["num_attention_heads"] #num_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Compression dimensions
        self.latent_dim = self.hidden_size // compression_ratio

        # A Matrix M is decomposed into D and U Matrix.
        # Compressed KV Projection.
        self.kv_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        # Compressed Q Projection
        self.q_proj_d = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        # Uncompress KQV Projections
        self.k_proj_u = nn.Linear(self.latent_dim, self.hidden_size // 2, bias=False)
        self.v_proj_u = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        self.q_proj_u = nn.Linear(self.latent_dim, self.hidden_size // 2, bias=False)
        # RoPE Key Components. K is built from X. Q is built from q_proj_d
        self.rope_k = nn.Linear(self.hidden_size, self.hidden_size // 2, bias=False)
        self.rope_q = nn.Linear(self.latent_dim, self.hidden_size // 2, bias=False)
        # Output Projection
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # RoPE Embeddings. Half size for RoPE components
        self.rotary_emb = rotary_emb #LlamaRotaryEmbedding(self.head_dim // 2)


    def forward(self, x, attention_mask = None):
        batch_size, seq_len, _ = x.shape

        #Compressed KV Projections
        kv_d = self.kv_proj_d(x) # [bs, seq Len, Latent_dim]
        if torch.isnan(kv_d).any():
            print("NaN detected in kv_d")

        #Compressed Q Projections
        q_d = self.q_proj_d(x) # [bs, seq Len, Latent_dim]
        if torch.isnan(q_d).any():
            print("NaN detected in q_d")

        #UnCompressed KV & Q Projections
        k_proj_2 = self.k_proj_u(kv_d) # [bs, seq_len, hidden size//2]
        if torch.isnan(k_proj_2).any():
            print("NaN detected in k_proj_2")

        q_proj_2 = self.q_proj_u(q_d) # [bs, seq_Len, hidden_size//2]
        if torch.isnan(q_proj_2).any():
            print("NaN detected in q_proj_2")
        v = self.v_proj_u(kv_d) # [ bs, seq len, hidden size]
        if torch.isnan(v).any():
            print("NaN detected in v")

        #Generate ROPE Components
        k_rope_2 = self.rope_k(x) # [bs, seq len, hidden_size//2]
        if torch.isnan(k_rope_2).any():
            print("NaN detected in k_rope_2")

        q_rope_2 = self.rope_q(q_d) # [bs, seq len, hidden_size//2]
        if torch.isnan(q_rope_2).any():
            print("NaN detected in q_rope_2")
        #Reshape components for heads before ROPE

        k_proj_2=k_proj_2.view(batch_size, seq_len, self.num_heads, self.head_dim//2)
        k_rope_2=k_rope_2.view(batch_size, seq_len, self.num_heads, self.head_dim//2)

        q_proj_2=q_proj_2.view(batch_size, seq_len, self.num_heads, self.head_dim//2)
        q_rope_2=q_rope_2.view(batch_size, seq_len, self.num_heads, self.head_dim//2)

        #Apply ROPE to positional aware components
        rotary_embeddings = self.rotary_emb.compute_rotary_embeddings(seq_len, x.device)
        k_rope_2 = self.rotary_emb.apply_rotary_emb(k_rope_2, rotary_embeddings)
        if torch.isnan(k_rope_2).any():
            print("NaN detected in k_rope_2 after ROPE")
        q_rope_2 = self.rotary_emb.apply_rotary_emb(q_rope_2, rotary_embeddings)
        if torch.isnan(q_rope_2).any():
            print("NaN detected in q_rope_2 after ROPE")

        # Concatenate KV vectors with KV ROPE vectors
        k = torch.cat([k_proj_2, k_rope_2], dim=-1)# [batch size, seq Len, num heads, head_dim]
        if torch.isnan(k).any():
            print("NaN detected in k")

        q = torch.cat([q_proj_2, q_rope_2], dim=-1)# [batch size, seq Len, num heads, head_dim]
        if torch.isnan(q).any():
            print("NaN detected in q")
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim) # [batch size, seq_Len, num_heads, head dim]
        if torch.isnan(v).any():
            print("NaN detected in v after view")

        #Reshape
        q = q.transpose(1, 2) # [batch size, num heads, seq Len, head_dim]
        if torch.isnan(q).any():
            print("NaN detected in q after transpose")

        k = k.transpose(1, 2) # [batch_size, num_heads, seq Len, head_dim]
        if torch.isnan(k).any():
            print("NaN detected in k after transpose")
        v = v.transpose(1, 2) # [batch_size, num_heads, seq Len, head_dim]
        if torch.isnan(v).any():
            print("NaN detected in v after transpose")

        #Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
                                                    q, k, v,
                                                    attn_mask=attention_mask,
                                                    dropout_p=0.0,
                                                    is_causal=True
        )# [batch size, num heads, seq Len, head dim]
        if torch.isnan(attn_output).any():
            print("NaN detected in attn_output")
        #Reshape and project output

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size) # [batch_size, seq_Len, hidden_size]
        if torch.isnan(attn_output).any():
            print("NaN detected in attn_output after view")
        return self.o_proj(attn_output) # [batch size, seq len, hidden_size]

class DeepSeekExpertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.intermediate_size = intermediate_size # Store for scaling
        self.scaling_factor = 8.0
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.down_proj.weight)

    def forward(self, x):
        gate_proj_out = self.gate_proj(x)
        # print(f"Rank {dist.get_rank()}: Gate Proj Output - NaN: {torch.isnan(gate_proj_out).any()}, Max: {gate_proj_out.max().item():.4f}, Min: {gate_proj_out.min().item():.4f}")
        act_fn_out = self.act_fn(gate_proj_out)
        # print(f"Rank {dist.get_rank()}: Act Fn Output - NaN: {torch.isnan(act_fn_out).any()}, Max: {act_fn_out.max().item():.4f}, Min: {act_fn_out.min().item():.4f}")
        up_proj_out = self.up_proj(x)
        # print(f"Rank {dist.get_rank()}: Up Proj Output - NaN: {torch.isnan(up_proj_out).any()}, Max: {up_proj_out.max().item():.4f}, Min: {up_proj_out.min().item():.4f}")
        intermediate = act_fn_out * up_proj_out
        # Scale down the intermediate product (IMP: Cause for NaN issue)
        intermediate = intermediate / self.scaling_factor
        # print(f"Rank {dist.get_rank()}: Intermediate Product - NaN: {torch.isnan(intermediate).any()}, Max: {intermediate.max().item():.4f}, Min: {intermediate.min().item():.4f}")
        down_proj_out = self.down_proj(intermediate)
        # print(f"Rank {dist.get_rank()}: Down Proj Output - NaN: {torch.isnan(down_proj_out).any()}, Max: {down_proj_out.max().item():.4f}, Min: {down_proj_out.min().item():.4f}")
        return down_proj_out


class DeepSeekMoE (nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts=8, num_shared_experts=1, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_experts - num_shared_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        #Shared experts
        self.shared_experts = nn.ModuleList([
            DeepSeekExpertLayer(hidden_size, intermediate_size)
            for _ in range(self.num_shared_experts)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            DeepSeekExpertLayer(hidden_size, intermediate_size)
            for _ in range(self.num_routed_experts)
        ])

        #Router components
        self.router = nn.Linear(hidden_size, self.num_routed_experts, bias=False)
        self.routing_bias= nn.Parameter(torch.zeros(self.num_routed_experts))
    
    def forward(self, x):
        # --- Debugging Print for Input to MoE ---
        # if dist.is_initialized():
        #     rank = dist.get_rank()
        #     print(f"Rank {rank}: MoE Input - NaN: {torch.isnan(x).any()}, Max: {x.max().item():.4f}, Min: {x.min().item():.4f}")
        # else:
        #     print(f"MoE Input - NaN: {torch.isnan(x).any()}, Max: {x.max().item():.4f}, Min: {x.min().item():.4f}")
        # # --- End Debugging Print for Input to MoE ---
        batch_size , seq_len , hidden_size = x.shape

        # Process through shared experts
        shared_output = sum ( expert ( x ) for expert in self.shared_experts )

        if self.num_shared_experts > 1:
            shared_output = shared_output / self.num_shared_experts # Average if multiple shared experts

        # Calculate routing scores
        routing_logits = self.router(x) + self.routing_bias

        #Get top-k experts per token
        routing_probs = torch.sigmoid(routing_logits) # Or F.softmax if you are testing that
        scores, indices = torch.topk(routing_probs, self.top_k, dim=-1)

        # --- Debugging Prints (Existing) ---
        # if dist.is_initialized():
        #     rank = dist.get_rank()
        #     print(f"Rank {rank}: MoE Routing Logits - NaN: {torch.isnan(routing_logits).any()}, Max: {routing_logits.max().item():.4f}, Min: {routing_logits.min().item():.4f}")
        #     print(f"Rank {rank}: MoE Routing Probs (Sigmoid) - NaN: {torch.isnan(routing_probs).any()}, Max: {routing_probs.max().item():.4f}, Min: {routing_probs.min().item():.4f}")
        #     print(f"Rank {rank}: MoE Top-K Scores - NaN: {torch.isnan(scores).any()}, Max: {scores.max().item():.4f}, Min: {scores.min().item():.4f}")
        #     # expert_usage = torch.bincount(indices.flatten(), minlength=self.num_routed_experts)
        #     # print(f"Rank {rank}: Expert Usage: {expert_usage}")
        # else:
        #     print(f"MoE Routing Logits - NaN: {torch.isnan(routing_logits).any()}, Max: {routing_logits.max().item():.4f}, Min: {routing_logits.min().item():.4f}")
        #     print(f"MoE Routing Probs (Sigmoid) - NaN: {torch.isnan(routing_probs).any()}, Max: {routing_probs.max().item():.4f}, Min: {routing_probs.min().item():.4f}")
        #     print(f"MoE Top-K Scores - NaN: {torch.isnan(scores).any()}, Max: {scores.max().item():.4f}, Min: {scores.min().item():.4f}")
        #     # expert_usage = torch.bincount(indices.flatten(), minlength=self.num_routed_experts)
            # print(f"Expert Usage: {expert_usage}")
        # --- End Debugging Prints (Existing) ---

        #Normalize the top-k scores
        scores = scores / scores.sum(dim=-1, keepdim=True)

        #Process through selected experts
        combined_output = torch.zeros_like(x)

        for k in range(self.top_k):
            expert_indices = indices[..., k]
            expert_scores = scores[..., k:k+1]

            # Process each expert
            for i in range(self.num_routed_experts):
                mask = (expert_indices == i)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.routed_experts[i](expert_input)

                    # --- Debugging Print for Expert Output ---
                    # if dist.is_initialized():
                    #     rank = dist.get_rank()
                    #     print(f"Rank {rank}: Expert {i} Output - NaN: {torch.isnan(expert_output).any()}, Max: {expert_output.max().item():.4f}, Min: {expert_output.min().item():.4f}")
                    # else:
                    #     print(f"Expert {i} Output - NaN: {torch.isnan(expert_output).any()}, Max: {expert_output.max().item():.4f}, Min: {expert_output.min().item():.4f}")
                    # # --- End Debugging Print for Expert Output ---

                    combined_output[mask] += expert_output * expert_scores[mask]

        # Combine shared and routed outputs
        final_output = shared_output + combined_output

        # --- Debugging Print for Final MoE Output ---
        # if dist.is_initialized():
        #     rank = dist.get_rank()
        #     print(f"Rank {rank}: Final MoE Output - NaN: {torch.isnan(final_output).any()}, Max: {final_output.max().item():.4f}, Min: {final_output.min().item():.4f}")
        # else:
        #     print(f"Final MoE Output - NaN: {torch.isnan(final_output).any()}, Max: {final_output.max().item():.4f}, Min: {final_output.min().item():.4f}")
        # --- End Debugging Print for Final MoE Output ---

        return final_output

        #Adjust bias terms based on expert Load
    def update_bias_terms(self, expert_load):
        target_load = 1.0 / self.num_routed_experts
        load_diff = expert_load - target_load

        # Dynamic update rate based on the magnitude of the load imbalance
        update_rate = 0.1 * torch.abs(load_diff)
        # Update the routing bias using the dynamic update rate
        self.routing_bias.data -= update_rate * load_diff
    

# class LlamaAttention(nn.Module):
#     """
#     (self_attn): LlamaAttention(
#           (q_proj): Linear(in_features=576, out_features=576, bias=False)
#           (k_proj): Linear(in_features=576, out_features=192, bias=False)
#           (v_proj): Linear(in_features=576, out_features=192, bias=False)
#           (o_proj): Linear(in_features=576, out_features=576, bias=False)
#     )
#     """
#     def __init__(self, config, rotary_emb):
#         super().__init__()
#         self.config = config
#         self.num_attention_heads = self.config['num_attention_heads']
#         self.hidden_size = self.config['hidden_size']
#         # Ensure the hidden size is divisible by the number of attention heads
#         if self.hidden_size % self.num_attention_heads != 0:
#             raise ValueError(
#                 f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
#             )
#         self.num_key_value_heads = self.config['num_key_value_heads']
#         self.head_dim =  self.hidden_size // self.num_attention_heads
#         self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # D,D
#         self.k_proj = nn.Linear(self.hidden_size, self.head_dim*self.num_key_value_heads, bias=False)   # D,D/H
#         self.v_proj = nn.Linear(self.hidden_size, self.head_dim*self.num_key_value_heads, bias=False)   # D,D/H
#         self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)   # D,D

#         # Convert the mask to boolean type when creating it
#         # self.register_buffer("mask", 
#         #                    torch.triu(torch.ones(self.config['max_position_embeddings'], 
#         #                                        self.config['max_position_embeddings']),
#         #                             diagonal=1))  # Convert to boolean
        
#         self.rotary_pos_emb = rotary_emb

#     def forward(self, x):
#         B, T, C = x.size()

#         q = self.q_proj(x)  # B,T,D
#         k = self.k_proj(x)  # B,T,D/H
#         v = self.v_proj(x)  # B,T,D/H

#         q = q.view(B, T, self.num_attention_heads, self.head_dim) # B,T,H,D
#         k = k.view(B, T, self.num_key_value_heads, self.head_dim) # B,T,H,D
#         v = v.view(B, T, self.num_key_value_heads, self.head_dim) # B,T,H,D

#         q = q.transpose(1,2) # B,H,T,D
#         k = k.transpose(1,2) # B,num_key_value_heads,T,D
#         v = v.transpose(1,2) # B,num_key_value_heads,T,D

#         # apply rotary positional embedding
#         q = self.rotary_pos_emb(q, T)
#         k = self.rotary_pos_emb(k, T)

#         # Repeat k/v heads if num_key_value_heads < num_attention_heads
#         if self.num_key_value_heads != self.num_attention_heads:
#             k = k.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1) # B,kv_head,T,D -> B,H,T,D
#             v = v.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=1) # B,kv_head,T,D -> B,H,T,D

#         # Manual attention Stats
#         # Q(B,H,T,D) @K.T(B,H,D,T) = Q.K_T (B,H,T,T)
#         # attn_scores = q @ k.transpose(-2,-1) # B,H,T,T
#         # mask_bool = self.mask[:T,:T].bool() # T,T
#         # attn_scores.masked_fill_(mask_bool, -torch.inf) # B,H,T,T
#         # attn_weights = F.softmax(attn_scores/k.size(-1)**0.5, dim=-1) # B,H,T,T
#         # context_vector = attn_weights @ v # B,H,T,T * B,H,T,D = B,H,T,D
#         # context_vector = context_vector.transpose(1,2) # B,T,H,D
#         # context_vector = context_vector.contiguous().view(B,T,C) # B,T,H,D -> B,T,D
#         # Manual attention Stats ENDS

#         # Scaled dot-product attention STARTS   
#         attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
#         context_vector = attn_out.transpose(1,2).reshape(B,T,C)
#         # Scaled dot-product attention ENDS

#         context_vector = self.o_proj(context_vector)
        
#         return context_vector

class LlamaMLP (nn.Module):
    def __init__(self, config, num_experts, num_shared_experts,
                 top_k):
        super().__init__()
        self.moe = DeepSeekMoE(
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k
        )

    def forward(self, x):
        return self.moe(x)
# class LlamaMLP(nn.Module):
#     """
#     (mlp): LlamaMLP(
#           (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
#           (up_proj): Linear(in_features=576, out_features=1536, bias=False)
#           (down_proj): Linear(in_features=1536, out_features=576, bias=False)
#           (act_fn): SiLU()
#         )
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.gate_proj = nn.Linear(self.config['hidden_size'], self.config['intermediate_size'], bias=False)
#         self.up_proj = nn.Linear(self.config['hidden_size'], self.config['intermediate_size'], bias=False)
#         self.down_proj = nn.Linear(self.config['intermediate_size'], self.config['hidden_size'], bias=False)
#         self.act_fn = SiLU()
#     def forward(self, x):
#         gate = self.gate_proj(x)
#         up = self.up_proj(x)
#         down = self.down_proj(self.act_fn(gate)*up)
#         return down 
    
class LlamaRMSNorm(nn.Module):
    """
    (norm): LlamaRMSNorm((576,), eps=1e-05)
        # RMSNorm Formula:
        #    RMS(x) = sqrt((1 / d) * sum(x_i^2 for i in range(d)))
        #    x_normalized = x / RMS(x)
        #    output = gamma * x_normalized
    
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eps = self.config['rms_norm_eps']
        self.weight = nn.Parameter(torch.ones(self.config['hidden_size']))
    def forward(self, x):
        rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return  self.weight *rms * x
    
# class LlamaDecoderLayer(nn.Module):
#     def __init__(self, config, rotary_emb):
#         super().__init__()
#         self.config = config
#         self.self_attn = LlamaAttention(self.config, rotary_emb)
#         self.mlp = LlamaMLP(self.config)
#         self.input_layernorm = LlamaRMSNorm(self.config)
#         self.post_attention_layernorm = LlamaRMSNorm(self.config)   
    
#     def forward(self, x):
#         residual = x
#         x = self.input_layernorm(x)
#         x = self.self_attn(x)
#         x = x + residual

#         residual = x
#         x = self.post_attention_layernorm(x)
#         x = self.mlp(x)
#         x = x + residual
#         return x 

class LlamaDecoderLayer (nn.Module):
    def __init__(self, config, rotary_emb, compression_ratio,
                 num_experts, num_shared_experts, top_k):
        super().__init__()
        self.self_attn = MultiHeadLatentAttention(rotary_emb, config, compression_ratio)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)
        self.mlp = LlamaMLP(config, num_experts, num_shared_experts,
                            top_k)

    def forward(self, x, attention_mask=None):
        # Self-attention
        residual = x
        x = self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + residual

        #Feedforward
        residual = x
        x = self.mlp(self.post_attention_layernorm(x))
        x = x + residual
        return x
    
class LlamaModel(nn.Module):
    def __init__(self, config, deepseek):
        super().__init__()
        self.init_method = config['init_method']
        self.config = config['model_config']
        self.deepseek = deepseek
        self.embed_tokens = nn.Embedding(self.config['vocab_size'], self.config['hidden_size'])
        # self.rotary_emb = RotaryPositionalEmbedding(self.config['hidden_size'], self.config['rope_theta'])
        num_heads = self.config['num_attention_heads']
        head_dim = self.config['hidden_size'] // num_heads
        self.rotary_emb = RotaryPositionalEmbedding(head_dim // 2, self.config['rope_theta'])
        # self.layers = nn.ModuleList([LlamaDecoderLayer(self.config, self.rotary_emb) for _ in range(self.config['num_hidden_layers'])])
        self.layers = nn.ModuleList([LlamaDecoderLayer(self.config, self.rotary_emb, deepseek["compression_ratio"], deepseek["num_experts"], deepseek["num_shared_experts"], deepseek["top_k"]) for _ in range(self.config['num_hidden_layers'])])
        self.norm = LlamaRMSNorm(self.config)
        self.lm_head = nn.Linear(self.config['hidden_size'], self.config['vocab_size'], bias=False)
        
        if self.config['tie_word_embeddings']:
            self.lm_head.weight = self.embed_tokens.weight
        
        self.apply(lambda m: _init_weights(m, self.init_method['std']))
    
    def forward(self, x, y=None):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x) # B,T,V
        logits = logits.view(-1, logits.size(-1))  # Shape: [B*T, V]
        if y is not None:
            y = y.view(-1)  # Shape: [B*T]
            loss = torch.nn.functional.cross_entropy(logits, y)
            return logits, loss
        else:
            return logits, None

    