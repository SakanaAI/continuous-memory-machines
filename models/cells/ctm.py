import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize

from models.cells.base import RecurrentCell
from models.cells.utils import Bounded


class SynapseMLP(nn.Module):
    def __init__(self, out_dims: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(out_dims), nn.Tanh(), nn.Linear(out_dims, out_dims), nn.LayerNorm(out_dims))

    def forward(self, x):
        return self.net(x)


class SynapseUNet(nn.Module):
    def __init__(self, out_dims: int, depth: int = 8, minimum_width: int = 16, dropout: float = 0.0):
        super().__init__()
        widths = np.linspace(out_dims, minimum_width, depth).astype(int)

        self.first_proj = nn.Sequential(nn.LazyLinear(int(widths[0])), nn.LayerNorm(int(widths[0])), nn.SiLU())

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.skip_norms = nn.ModuleList()
        for i in range(len(widths) - 1):
            self.down.append(nn.Sequential(nn.Dropout(dropout), nn.Linear(int(widths[i]), int(widths[i + 1])), nn.LayerNorm(int(widths[i + 1])), nn.SiLU()))
            self.up.append(nn.Sequential(nn.Dropout(dropout), nn.Linear(int(widths[i + 1]), int(widths[i])), nn.LayerNorm(int(widths[i])), nn.SiLU()))
            self.skip_norms.append(nn.LayerNorm(int(widths[i])))

    def forward(self, x):
        outs = [self.first_proj(x)]
        for layer in self.down:
            outs.append(layer(outs[-1]))

        h = outs[-1]
        n = len(self.up)
        for i in range(n):
            idx = n - 1 - i
            h = self.skip_norms[idx](self.up[idx](h) + outs[idx])
        return h


def construct_synapse(d_model: int, synapse_type: str = "mlp", synapse_depth: int = 8, dropout: float = 0.0) -> nn.Module:
    if synapse_type == "mlp":
        return SynapseMLP(out_dims=d_model, dropout=dropout)
    elif synapse_type == "unet":
        return SynapseUNet(out_dims=d_model, depth=synapse_depth, dropout=dropout)
    raise ValueError(f"Unknown synapse type: {synapse_type}")


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class SuperLinear(nn.Module):
    def __init__(self, in_dims, out_dims, N, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.in_dims = in_dims
        self.register_parameter("w1", nn.Parameter(torch.empty((in_dims, out_dims, N))))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w1)
        # nn.init.zeros_(self.w1)
        # with torch.no_grad():
        #     self.w1.data[-1, 0, :] = 2.0

    def forward(self, x):
        out = self.dropout(x)
        out = torch.einsum("BDM,MHD->BDH", out, self.w1)
        return out.squeeze(-1)


class Synchronizer(nn.Module):
    def __init__(self, d_model: int, sync_repr_size: int):
        super().__init__()
        self.sync_repr_size = sync_repr_size
        self._set_left_right_neuron_idxs(d_model, sync_repr_size)
        self.r = nn.Parameter(torch.empty(sync_repr_size))
        self.rmsn = nn.RMSNorm(sync_repr_size, elementwise_affine=True)
        self.reset_parameters()

    def reset_parameters(self):
        if parametrize.is_parametrized(self, "r"):
            parametrize.remove_parametrizations(self, "r", leave_parametrized=False)
        nn.init.uniform_(self.r, 0, 1)
        parametrize.register_parametrization(self, "r", Bounded(0, 15))
        self.rmsn.reset_parameters()

    def _set_left_right_neuron_idxs(self, d_model: int, sync_repr_size: int):
        total_pairs = d_model**2
        assert sync_repr_size <= total_pairs, f"Sync repr size ({sync_repr_size}) > total pairs ({total_pairs})"
        pair_idxs = torch.randperm(total_pairs)[:sync_repr_size].sort()[0]
        self.neuron_idxs_left, self.neuron_idxs_right = nn.Buffer(pair_idxs // d_model), nn.Buffer(pair_idxs % d_model)

    def forward(self, z: Tensor, alpha_0: Tensor, beta_0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z_z = z[:, self.neuron_idxs_left] * z[:, self.neuron_idxs_right]
        exp_minus_r = torch.exp(-self.r).unsqueeze(0).repeat(z.shape[0], 1)
        alpha_1, beta_1 = exp_minus_r * alpha_0 + z_z, exp_minus_r * beta_0 + 1  # α_t = e^-r⋅α_t-1 + z_i_t ⊗ z_j_t; β_t = e^-r⋅β_t-1 + 1
        S = self.rmsn(alpha_1 / beta_1 + 1e-8)  # S_t = α_t/β_t
        return S, alpha_1, beta_1

    def initialize_state(self, activated_state):
        decay_alpha = torch.zeros(1, self.sync_repr_size, device=activated_state.device)
        decay_beta = torch.zeros(1, self.sync_repr_size, device=activated_state.device)
        initial_synch = activated_state[:, self.neuron_idxs_left] * activated_state[:, self.neuron_idxs_right]
        return decay_alpha, decay_beta, initial_synch


class ContinuousThoughtMachine(RecurrentCell):
    def __init__(self, d_input, d_model, memory_length, dropout, synapse_type="mlp", synapse_depth=8):
        super().__init__(d_input=d_input, d_model=d_model)
        self.memory_length = memory_length
        self.synapses = construct_synapse(d_model, synapse_type=synapse_type, synapse_depth=synapse_depth, dropout=dropout)
        self.nlms = nn.Sequential(SuperLinear(in_dims=memory_length, out_dims=2, N=d_model, dropout=dropout), nn.GLU(), Squeeze(-1))
        self.synchronizer = Synchronizer(d_model=d_model, sync_repr_size=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.start_stm = nn.Parameter(torch.empty((d_model, memory_length)))  # (H, M)
        self.start_activated_state = nn.Parameter(torch.empty((d_model)))  # (H,)
        self.reset_parameters()

    def reset_parameters(self):
        self.synchronizer.reset_parameters()
        self.nlms[0].reset_parameters()
        nn.init.zeros_(self.start_stm)
        nn.init.zeros_(self.start_activated_state)

    def initialize_state(self):
        stm, activated_state = self.start_stm.unsqueeze(0), self.start_activated_state.unsqueeze(0)  # (1, H, M), (1, H)
        decay_alpha, decay_beta, initial_synch = self.synchronizer.initialize_state(activated_state)
        return (activated_state, stm, decay_alpha, decay_beta, initial_synch)

    def forward(self, input, state):
        assert input.ndim == 2, "Input should be of size B x D"
        post_activated_state, stm, decay_alpha, decay_beta, _ = self.dropout(state[0]), state[1], state[2], state[3], state[4]

        # --- Apply Synapses ---
        pre_activated_state = self.synapses(torch.concat([input, post_activated_state], dim=-1))
        stm = torch.cat((stm[:, :, 1:], pre_activated_state.unsqueeze(-1)), dim=-1)

        # --- Apply Neuron-Level Models ---
        post_activated_state = self.nlms(stm)

        # --- Prediction Synchronization ---
        synchronisation, decay_alpha, decay_beta = self.synchronizer(post_activated_state, decay_alpha, decay_beta)
        return synchronisation, (post_activated_state, stm, decay_alpha, decay_beta, synchronisation)


class CrossAttentionContinuousThoughtMachine(ContinuousThoughtMachine):
    def __init__(self, d_input, d_model, memory_length, dropout, synapse_type="mlp", synapse_depth=8):
        super().__init__(d_input, d_model, memory_length, dropout, synapse_type, synapse_depth)
        self.input_synchronizer = Synchronizer(d_model=d_model, sync_repr_size=d_model)

    def initialize_state(self):
        activated_state, stm, decay_alpha, decay_beta, initial_synch = super().initialize_state()
        decay_alpha_input, decay_beta_input, synchronization_input = self.input_synchronizer.initialize_state(activated_state)
        return (activated_state, stm, decay_alpha, decay_beta, initial_synch, decay_alpha_input, decay_beta_input, synchronization_input)

    def forward(self, input, state):
        assert input.ndim == 2, "Input should be of size B x D"
        activated_state, stm, decay_alpha, decay_beta, synchronization, decay_alpha_input, decay_beta_input, synchronization_input = state
        synchronisation, (activated_state, stm, decay_alpha, decay_beta, synchronization) = super().forward(input, (activated_state, stm, decay_alpha, decay_beta, synchronization))

        # --- Input Synchronization ---
        synchronization_input, decay_alpha_input, decay_beta_input = self.input_synchronizer(activated_state, decay_alpha_input, decay_beta_input)

        return synchronisation, (activated_state, stm, decay_alpha, decay_beta, synchronization, decay_alpha_input, decay_beta_input, synchronization_input)
