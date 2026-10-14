import torch

from models.cells.cmm import MemoryUpdater
from models.cells.ctm import ContinuousThoughtMachine, Synchronizer


class CTMTransformerCell(ContinuousThoughtMachine):
    """CTM ablation with a Transformer over its sliding pre-activation window.

    Unlike the CMM, this cell has no persistent long-term-memory tokens. The
    Transformer only mixes the current CTM short-term-memory window before the
    standard neuron-level models consume it.
    """

    def __init__(
        self,
        d_input,
        d_model,
        memory_length,
        dropout,
        memory_transformer_layers=1,
        num_heads=4,
        synapse_type="mlp",
        synapse_depth=8,
        memory_update_gate=False,
        memory_update_gate_init=0.0,
    ):
        super().__init__(
            d_input=d_input,
            d_model=d_model,
            memory_length=memory_length,
            dropout=dropout,
            synapse_type=synapse_type,
            synapse_depth=synapse_depth,
        )
        self.memory_updater = MemoryUpdater(
            d_model=d_model,
            num_ltm_tokens=0,
            num_stm_tokens=memory_length,
            num_layers=memory_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            memory_update_gate=memory_update_gate,
            memory_update_gate_init=memory_update_gate_init,
        )

    def forward(self, input, state):
        assert input.ndim == 2, "Input should be of size B x D"
        post_activated_state, stm, decay_alpha, decay_beta, _ = (
            self.dropout(state[0]),
            state[1],
            state[2],
            state[3],
            state[4],
        )

        pre_activated_state = self.synapses(
            torch.concat([input, post_activated_state], dim=-1)
        )
        stm = torch.cat(
            (stm[:, :, 1:], pre_activated_state.unsqueeze(-1)), dim=-1
        )

        empty_ltm = stm.new_empty(stm.shape[0], 0, self.d_model)
        transformed_stm, _ = self.memory_updater(stm.transpose(1, 2), empty_ltm)
        post_activated_state = self.nlms(transformed_stm.transpose(1, 2))

        synchronisation, decay_alpha, decay_beta = self.synchronizer(
            post_activated_state, decay_alpha, decay_beta
        )
        return synchronisation, (
            post_activated_state,
            stm,
            decay_alpha,
            decay_beta,
            synchronisation,
        )


class CrossAttentionCTMTransformerCell(CTMTransformerCell):
    def __init__(
        self,
        d_input,
        d_model,
        memory_length,
        dropout,
        memory_transformer_layers=1,
        num_heads=4,
        synapse_type="mlp",
        synapse_depth=8,
        memory_update_gate=False,
        memory_update_gate_init=0.0,
    ):
        super().__init__(
            d_input=d_input,
            d_model=d_model,
            memory_length=memory_length,
            dropout=dropout,
            memory_transformer_layers=memory_transformer_layers,
            num_heads=num_heads,
            synapse_type=synapse_type,
            synapse_depth=synapse_depth,
            memory_update_gate=memory_update_gate,
            memory_update_gate_init=memory_update_gate_init,
        )
        self.input_synchronizer = Synchronizer(
            d_model=d_model, sync_repr_size=d_model
        )

    def initialize_state(self):
        activated_state, stm, decay_alpha, decay_beta, initial_synch = (
            super().initialize_state()
        )
        decay_alpha_input, decay_beta_input, synchronization_input = (
            self.input_synchronizer.initialize_state(activated_state)
        )
        return (
            activated_state,
            stm,
            decay_alpha,
            decay_beta,
            initial_synch,
            decay_alpha_input,
            decay_beta_input,
            synchronization_input,
        )

    def forward(self, input, state):
        (
            activated_state,
            stm,
            decay_alpha,
            decay_beta,
            synchronization,
            decay_alpha_input,
            decay_beta_input,
            synchronization_input,
        ) = state
        synchronisation, base_state = super().forward(
            input,
            (activated_state, stm, decay_alpha, decay_beta, synchronization),
        )
        activated_state, stm, decay_alpha, decay_beta, synchronization = base_state
        synchronization_input, decay_alpha_input, decay_beta_input = (
            self.input_synchronizer(
                activated_state, decay_alpha_input, decay_beta_input
            )
        )
        return synchronisation, (
            activated_state,
            stm,
            decay_alpha,
            decay_beta,
            synchronization,
            decay_alpha_input,
            decay_beta_input,
            synchronization_input,
        )
