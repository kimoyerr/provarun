import math
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from provarun.model_utils import apply_rotary_emb, precompute_pos_cis


# TODO: Implement Gaussian Fourier Transform from https://github.com/HannesStark/dirichlet-flow-matching/blob/main/model/dna_models.py
# Time step embedding: From https://github.com/andrew-cr/discrete_flow_models/blob/main/flow_model.py which is from
# https://github.com/yang-song/score_sde_pytorch/ which is from
# https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # assumes timesteps is in the range 0 to 1000

    assert len(timesteps.shape) == 1  or timesteps.shape[1] == 1
    timesteps.squeeze_(1)
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


# The transformer model is inspired by Minimind: https://github.com/jingyaogong/minimind/blob/master/model/model.py and others
class MHA(nn.Module):
    def __init__(self, config, pos_cis):
        super().__init__()
        self.config = config
        self.head_dim = self.config.dim // self.config.num_heads
        self.num_heads = self.config.num_heads
        
        # Linear layers for the query, key, and value
        # TODO: Implement Key and Value cache for faster inference
        # TODO: Implement more heads for Key and Value (multi-query attention: https://arxiv.org/pdf/2305.13245). Also implemented https://github.com/jingyaogong/minimind/blob/master/model/model.py
        self.wq = nn.Linear(self.config.dim, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.config.dim, self.num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.config.dim, self.num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.config.dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # Flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

        # Set causal mask
        causal_mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1)
        # Convert to -inf
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x, pos_cis):
        batch_size, seq_len, _ = x.size()
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # Expand the dimensions of the query, key, and value
        xq = xq.view(x.size(0), x.size(1), self.config.num_heads, self.head_dim)
        xk = xk.view(x.size(0), x.size(1), self.config.num_heads, self.head_dim)
        xv = xv.view(x.size(0), x.size(1), self.config.num_heads, self.head_dim)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # Transpose the query, key, and value to make them compatible with the attention function
        # This moves the head dimension to the second position and we caclulate the attention across the sequence length for each head
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash:
            # TODO: Implement Flash attention
            raise NotImplementedError("Flash attention is not implemented yet.")
        else:
            # This gives the attention scores for each head and at each position
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / (self.head_dim ** 0.5) # Returns [batch_size, num_heads, seq_len, seq_len]
            # Apply causal masking for regular training, skip for flow matching
            if self.config.flow_matching:
                scores = scores
            else:
                scores = scores + self.causal_mask[:, :, :, :seq_len]
            scores = nn.functional.softmax(scores, dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # Returns [batch_size, num_heads, seq_len, head_dim]
        
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, -1)  # Returns [batch_size, seq_len, dim] which is the original shape of input x
        output= self.wo(output) # Combine the heads to get the final output using the output linear layer
        output = self.resid_dropout(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # TODO: Are there better ways to do this?
        hidden_dim = 4 * config.dim 
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        output = self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
        output = self.dropout(output)
        return output



class TransformerBlock(nn.Module):
    def __init__(self, config, pos_cis):
        super().__init__()
        self.config = config
        self.attn = MHA(config, pos_cis)
        self.attn_norm = nn.RMSNorm(config.dim, config.rms_norm_eps)
        self.ffn = FeedForward(config)
        self.ffn_norm = nn.RMSNorm(config.dim, config.rms_norm_eps)
    
    def forward(self, x, pos_cis):
        h = x + self.attn(self.attn_norm(x), pos_cis)
        out = h + self.ffn(self.ffn_norm(h))

        return out
    

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)

        self.wte = nn.Embedding(config.vocab_size, config.dim)

        # Positional embeddings
        pos_cis = precompute_pos_cis(self.config.dim // self.config.num_heads, self.config.max_seq_len)
        self.register_buffer("pos_cis", pos_cis, persistent=False)

        # Number of transformer layers
        self.num_layers = config.num_transformer_layers
        self.layers = nn.ModuleList()
        for ll in range(self.num_layers):
            self.layers.append(TransformerBlock(config, pos_cis))
        
        self.norm = nn.RMSNorm(config.dim, config.rms_norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.OUT = CausalLMOutputWithPast()

        # Weight initilization
        self.apply(self._init_weights)

        # Special weight initialization for the output layers of feedforward and MHA
        # TODO: Other ways to initialize the weights? What's the best way for each layer in the transformer?
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_transformer_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        

    def forward(self, x, labels=None, times=None):
        batch_size, seq_len = x.size()
        current_idx = 0  # TODO: Implement sliding window for long sequences
        h = self.wte(x)  # Shape [batch_size, seq_len, dim]
        pos_cis = self.pos_cis[current_idx:current_idx + seq_len]
        # Add time embeddings if exists
        if times is not None:
            time_emb = transformer_timestep_embedding(times, self.config.dim)
            h = h + time_emb.view(-1, 1, self.config.dim)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h, pos_cis)
        
        h = self.norm(h)
        
        # Check if labels or targets are provided
        if labels is not None:
            logits = self.output(h)
            self.last_loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1, reduction="none")
        else:
            logits = self.output(h[:, [-1], :])  # Ignore the last token
            self.last_loss = None

        # Causal LM output
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("past_key_values", None)
        self.OUT.__setitem__("last_loss", self.last_loss)       

        return self.OUT



