import torch
from torch import nn

from models.cells.base import RecurrentCell
from models.cells.ctm import Synchronizer, SuperLinear, Squeeze, construct_synapse


class MemoryTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, memory_update_gate=False, memory_update_gate_init=0.0):
        super().__init__()
        self.num_heads, self.d_head = num_heads, d_model // num_heads
        self.scale = self.d_head ** -0.5
        self.memory_update_gate = nn.Parameter(torch.tensor(float(memory_update_gate_init))) if memory_update_gate else None
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model, bias=False), nn.SiLU(), nn.Linear(d_model, d_model, bias=False))
        self.dropout = nn.Dropout(dropout)
        self._capture_attention_statistics = False
        self._attention_sum = None
        self._attention_count = 0

    def start_attention_capture(self):
        """Accumulate attention averaged over batches and recurrent timesteps."""
        self._capture_attention_statistics = True
        self._attention_sum = None
        self._attention_count = 0

    def stop_attention_capture(self):
        self._capture_attention_statistics = False

    def mean_captured_attention(self):
        if self._attention_sum is None or self._attention_count == 0:
            return None
        return self._attention_sum / self._attention_count

    def forward(self, x_content, pos_emb):
        B, S, D = x_content.shape
        H, d_h = self.num_heads, self.d_head

        normed = self.attn_norm(x_content)
        normed_with_pe = normed + pos_emb  # PE for addressing
        
        q = self.q_proj(normed_with_pe).reshape(B, S, H, d_h).transpose(1, 2)  # (B, S, d_model) -> (B, H, S, d_h)
        k = self.k_proj(normed_with_pe).reshape(B, S, H, d_h).transpose(1, 2)  # (B, S, d_model) -> (B, H, S, d_h)
        v = self.v_proj(normed).reshape(B, S, H, d_h).transpose(1, 2)  # (B, S, d_model) -> (B, H, S, d_h)
        v = v.clone()
        v[:, :, 0, :] = 0  # sink

        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        self._last_attn = attn.detach()
        if self._capture_attention_statistics:
            batch_mean = self._last_attn.float().mean(dim=0)
            if self._attention_sum is None:
                self._attention_sum = batch_mean
            else:
                self._attention_sum = self._attention_sum + batch_mean
            self._attention_count += 1
        attn = self.dropout(attn)
        out = self.out_proj((attn @ v).transpose(1, 2).reshape(B, S, D))  # (B, H, S, d_h) -> (B, S, d_model)
        attn_out = self.dropout(out)
        if self.memory_update_gate is not None:
            attn_out = self.memory_update_gate * attn_out
        x_content = x_content + attn_out
        ffn_out = self.dropout(self.ffn(self.ffn_norm(x_content)))
        if self.memory_update_gate is not None:
            ffn_out = self.memory_update_gate * ffn_out
        return x_content + ffn_out


class MemoryUpdater(nn.Module):
    def __init__(self, d_model, num_ltm_tokens, num_stm_tokens, num_layers, num_heads=4, dropout=0.0, memory_update_gate=False, memory_update_gate_init=0.0):
        super().__init__()
        self.num_ltm_tokens = num_ltm_tokens
        self.sink = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([MemoryTransformerBlock(d_model, num_heads, dropout, memory_update_gate=memory_update_gate, memory_update_gate_init=memory_update_gate_init) for _ in range(num_layers)])
        self.ltm_norm = nn.LayerNorm(d_model) if num_ltm_tokens > 0 else nn.Identity()
        self.exclude_ltm_tokens = False

        total_len = 1 + num_ltm_tokens + num_stm_tokens
        self.pos_emb = nn.Parameter(torch.randn(1, total_len, d_model) * 0.02)

    def forward(self, stm, ltm):
        B = stm.shape[0]
        sink = self.sink.expand(B, -1, -1)
        if self.exclude_ltm_tokens:
            x = torch.cat([sink, stm], dim=1)
            pos_emb = torch.cat(
                [
                    self.pos_emb[:, :1],
                    self.pos_emb[:, 1 + self.num_ltm_tokens:],
                ],
                dim=1,
            )
        else:
            x = torch.cat([sink, ltm, stm], dim=1)  # (B, 1+L+M, d_model)
            pos_emb = self.pos_emb[:, :x.shape[1]]

        for block in self.blocks:
            x = block(x, pos_emb)  # PE passed separately

        if self.exclude_ltm_tokens:
            return x[:, 1:], ltm

        ltm_start = 1
        ltm = self.ltm_norm(x[:, ltm_start:ltm_start + self.num_ltm_tokens])
        stm = x[:, ltm_start + self.num_ltm_tokens:]
        return stm, ltm

class ContinuousMemoryMachine(RecurrentCell):
    def __init__(self, d_input, d_model, memory_length, dropout, num_long_term_memory_tokens=16, memory_transformer_layers=1, num_heads=4, synapse_type="mlp", synapse_depth=8, memory_update_gate=False, memory_update_gate_init=0.0):
        super().__init__(d_input=d_input, d_model=d_model)
        self.memory_length, self.num_long_term_memory_tokens = memory_length, num_long_term_memory_tokens

        self.synapses = construct_synapse(d_model, synapse_type=synapse_type, synapse_depth=synapse_depth, dropout=dropout)
        self.memory_updater = MemoryUpdater(d_model, num_ltm_tokens=num_long_term_memory_tokens, num_stm_tokens=memory_length, num_layers=memory_transformer_layers, num_heads=num_heads, dropout=dropout, memory_update_gate=memory_update_gate, memory_update_gate_init=memory_update_gate_init)

        self.register_buffer("start_activated_state", torch.zeros(d_model)) # (d_model,)
        self.start_ltm = nn.Parameter(torch.randn(num_long_term_memory_tokens, d_model) * 0.01) # (L, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.nlms = nn.Sequential(SuperLinear(in_dims=memory_length, out_dims=2, N=d_model, dropout=dropout), nn.GLU(), Squeeze(-1), nn.LayerNorm(d_model))
        self.synchronizer = Synchronizer(d_model=d_model, sync_repr_size=d_model)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.start_activated_state)
        nn.init.normal_(self.nlms[0].w1, std=0.02)

    def initialize_state(self):
        post_activated_state = self.start_activated_state.unsqueeze(0) # (1, d_model)
        stm = torch.zeros(1, self.d_model, self.memory_length, device=post_activated_state.device) # (1, d_model, M)
        ltm = self.start_ltm.unsqueeze(0) # (1, L, d_model)
        decay_alpha, decay_beta, initial_synch = self.synchronizer.initialize_state(post_activated_state)
        return (post_activated_state, stm, ltm, decay_alpha, decay_beta, initial_synch)

    def forward(self, input, state):
        assert input.ndim == 2, "Input should be of size B x D"
        post_activated_state, stm, ltm, decay_alpha, decay_beta, _ = self.dropout(state[0]), state[1], state[2], state[3], state[4], state[5]

        # --- Apply Synapse ---
        pre_activated_state = self.synapses(torch.concat([input, post_activated_state], dim=-1)) # (B, d_input + d_model) -> (B, d_model)

        # --- Update STM ---
        stm = torch.cat((stm[:, :, 1:], pre_activated_state.unsqueeze(-1)), dim=-1) # (B, d_model, M)

        # --- Update Memory ---
        stm_out, ltm = self.memory_updater(stm.transpose(1, 2), ltm) # stm_out: (B, M, d_model), ltm: (B, L, d_model)
        stm_out = stm_out.transpose(1, 2) # (B, d_model, M)

        # --- Apply Neuron-Level Models ---
        post_activated_state = self.nlms(stm_out) # (B, d_model, M) -> (B, d_model)

        # --- Prediction Synchronization ---
        synchronisation, decay_alpha, decay_beta = self.synchronizer(post_activated_state, decay_alpha, decay_beta)
        return synchronisation, (post_activated_state, stm, ltm, decay_alpha, decay_beta, synchronisation)


class CrossAttentionContinuousMemoryMachine(ContinuousMemoryMachine):
    def __init__(self, d_input, d_model, memory_length, dropout, num_long_term_memory_tokens=16, memory_transformer_layers=1, num_heads=4, synapse_type="mlp", synapse_depth=8, memory_update_gate=False, memory_update_gate_init=0.0):
        super().__init__(d_input, d_model, memory_length, dropout, num_long_term_memory_tokens, memory_transformer_layers, num_heads, synapse_type, synapse_depth, memory_update_gate, memory_update_gate_init)
        self.input_synchronizer = Synchronizer(d_model=d_model, sync_repr_size=d_model)

    def initialize_state(self):
        post_activated_state, stm, ltm, decay_alpha, decay_beta, initial_synch = super().initialize_state()
        decay_alpha_input, decay_beta_input, synchronization_input = self.input_synchronizer.initialize_state(post_activated_state)
        return (post_activated_state, stm, ltm, decay_alpha, decay_beta, initial_synch, decay_alpha_input, decay_beta_input, synchronization_input)

    def forward(self, input, state):
        assert input.ndim == 2, "Input should be of size B x D"
        post_activated_state, stm, ltm, decay_alpha, decay_beta, synchronization, decay_alpha_input, decay_beta_input, synchronization_input = state
        synchronisation, (post_activated_state, stm, ltm, decay_alpha, decay_beta, synchronization) = super().forward(input, (post_activated_state, stm, ltm, decay_alpha, decay_beta, synchronization))

        # --- Input Synchronization ---
        synchronization_input, decay_alpha_input, decay_beta_input = self.input_synchronizer(post_activated_state, decay_alpha_input, decay_beta_input)

        return synchronisation, (post_activated_state, stm, ltm, decay_alpha, decay_beta, synchronization, decay_alpha_input, decay_beta_input, synchronization_input)
