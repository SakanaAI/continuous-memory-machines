import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.cells.base import RecurrentCell
from models.cells.lstm import LSTMCell
from models.cells.rnn import RNNCell


_CONTROLLERS = {"LSTM": LSTMCell, "RNN": RNNCell}


class DifferentiableNeuralComputerCell(RecurrentCell):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        controller_type: str = "LSTM",
        memory_n: int = 128,
        memory_m: int = 20,
        num_read_heads: int = 1,
        num_write_heads: int = 1,
        dropout: float = 0.0,
        clip_value: float = 20.0,
    ):
        super().__init__(d_input=d_input, d_model=d_model)
        controller_size = d_model
        self.memory_n, self.memory_m = memory_n, memory_m
        self.num_read_heads, self.num_write_heads = num_read_heads, num_write_heads
        self.clip_value = clip_value

        # --- Controller ---
        ctrl_cls = _CONTROLLERS[controller_type]
        self.controller = ctrl_cls(
            d_input=d_input + num_read_heads * memory_m,
            d_model=controller_size,
            dropout=dropout,
        )
        self._ctrl_state_size = len(self.controller.initialize_state())

        # --- Interface Vector Projection (single linear, split per head) ---
        self.num_read_modes = 1 + 2 * num_write_heads
        self._read_param_size = (
            memory_m + 1 + 1 + self.num_read_modes
        )  # key(M), strength(1), free_gate(1), modes(1 + 2W)
        self._write_param_size = (
            3 * memory_m + 3
        )  # key(M), strength(1), erase(M), write_vec(M), alloc_gate(1), write_gate(1)
        total_param_size = (
            num_read_heads * self._read_param_size
            + num_write_heads * self._write_param_size
        )
        self.fc_interface = nn.Linear(controller_size, total_param_size)

        # --- Output Projection ---
        self.fc_output = nn.Linear(controller_size + num_read_heads * memory_m, d_model)

        # --- Memory (constant init, NOT learned) ---
        self.register_buffer("mem_bias", torch.full((memory_n, memory_m), 1e-6))

        # --- Link matrix diagonal mask (for zeroing self-loops) ---
        self.register_buffer("link_diag_mask", 1 - torch.eye(memory_n))

        # --- Learned Initial States ---
        self.read_biases = nn.ParameterList(
            [nn.Parameter(torch.empty(memory_m)) for _ in range(num_read_heads)]
        )
        self.read_weight_biases = nn.ParameterList(
            [nn.Parameter(torch.empty(memory_n)) for _ in range(num_read_heads)]
        )
        self.write_weight_biases = nn.ParameterList(
            [nn.Parameter(torch.empty(memory_n)) for _ in range(num_write_heads)]
        )

        # --- State unpacking indices ---
        self._idx_mem = self._ctrl_state_size
        self._idx_read_vecs = self._idx_mem + 1
        self._idx_read_w = self._idx_read_vecs + num_read_heads
        self._idx_write_w = self._idx_read_w + num_read_heads
        self._idx_link = self._idx_write_w + num_write_heads
        self._idx_prec = self._idx_link + 1
        self._idx_usage = self._idx_prec + 1

        self.dropout_layer = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for rb in self.read_biases:
            nn.init.normal_(rb, std=0.01)
        for wb in self.read_weight_biases:
            nn.init.normal_(wb, std=0.01)
        for wb in self.write_weight_biases:
            nn.init.normal_(wb, std=0.01)
        nn.init.xavier_uniform_(self.fc_interface.weight, gain=1.4)
        nn.init.normal_(self.fc_interface.bias, std=0.01)
        nn.init.xavier_uniform_(self.fc_output.weight, gain=1)
        nn.init.normal_(self.fc_output.bias, std=0.01)

    def initialize_state(self):
        ctrl_state = (
            self.controller.initialize_state()
        )  # tuple of (1, controller_size) tensors
        memory = self.mem_bias.unsqueeze(0)  # (1, N, M)
        read_vecs = tuple(
            torch.tanh(rb).unsqueeze(0) for rb in self.read_biases
        )  # R x (1, M)
        read_weights = tuple(
            F.softmax(wb, dim=0).unsqueeze(0) for wb in self.read_weight_biases
        )  # R x (1, N)
        write_weights = tuple(
            F.softmax(wb, dim=0).unsqueeze(0) for wb in self.write_weight_biases
        )  # W x (1, N)
        link_matrix = torch.zeros(
            1,
            self.num_write_heads,
            self.memory_n,
            self.memory_n,
            device=self.mem_bias.device,
        )  # (1, W, N, N)
        precedence = torch.zeros(
            1, self.num_write_heads, self.memory_n, device=self.mem_bias.device
        )  # (1, W, N)
        usage = torch.zeros(1, self.memory_n, device=self.mem_bias.device)  # (1, N)
        return (
            *ctrl_state,
            memory,
            *read_vecs,
            *read_weights,
            *write_weights,
            link_matrix,
            precedence,
            usage,
        )

    def _content_address(self, memory, key, strength):
        # memory: (B, N, M), key: (B, M), strength: (B,)
        key = key.unsqueeze(1)  # (B, 1, M)
        sim = F.cosine_similarity(memory + 1e-16, key + 1e-16, dim=-1)  # (B, N)
        return F.softmax(strength.unsqueeze(1) * sim, dim=1)  # (B, N)

    def _allocation(self, usage):
        # usage: (B, N) -> allocation weights: (B, N)
        eps = 1e-6
        usage = eps + (1 - eps) * usage  # (B, N)
        nonusage = 1 - usage  # (B, N)
        sorted_nonusage, indices = torch.topk(
            nonusage, k=self.memory_n, dim=1
        )  # (B, N)
        sorted_usage = 1 - sorted_nonusage  # (B, N)
        prod = torch.cumprod(sorted_usage, dim=1)  # (B, N)
        exclusive_prod = torch.cat(
            [torch.ones_like(prod[:, :1]), prod[:, :-1]], dim=1
        )  # (B, N)
        sorted_alloc = sorted_nonusage * exclusive_prod  # (B, N)
        _, inv_indices = torch.topk(
            indices, k=self.memory_n, dim=1, largest=False
        )  # (B, N)
        return sorted_alloc.gather(1, inv_indices)  # (B, N)

    def _update_link(self, prev_link, prev_precedence, write_weights):
        # prev_link: (B, W, N, N), prev_precedence: (B, W, N), write_weights: (B, W, N)
        w_i = write_weights.unsqueeze(3)  # (B, W, N, 1)
        w_j = write_weights.unsqueeze(2)  # (B, W, 1, N)
        p_j = prev_precedence.unsqueeze(2)  # (B, W, 1, N)
        link = (1 - w_i - w_j) * prev_link + w_i * p_j  # (B, W, N, N)
        return link * self.link_diag_mask  # (B, W, N, N)

    def _update_precedence(self, prev_precedence, write_weights):
        # prev_precedence: (B, W, N), write_weights: (B, W, N)
        write_sum = write_weights.sum(dim=-1, keepdim=True)  # (B, W, 1)
        return (1 - write_sum) * prev_precedence + write_weights  # (B, W, N)

    def forward(self, input: Tensor, state):
        assert input.ndim == 2, "Input should be of size B x D"
        N, M, R, W = (
            self.memory_n,
            self.memory_m,
            self.num_read_heads,
            self.num_write_heads,
        )

        # --- Unpack State ---
        ctrl_state = state[: self._ctrl_state_size]
        memory = state[self._idx_mem]  # (B, N, M)
        read_vecs = list(state[self._idx_read_vecs : self._idx_read_w])  # R x (B, M)
        read_weights = list(state[self._idx_read_w : self._idx_write_w])  # R x (B, N)
        write_weights = list(state[self._idx_write_w : self._idx_link])  # W x (B, N)
        link_matrix = state[self._idx_link]  # (B, W, N, N)
        precedence = state[self._idx_prec]  # (B, W, N)
        usage = state[self._idx_usage]  # (B, N)

        # --- Controller ---
        controller_input = torch.cat([input] + read_vecs, dim=1)  # (B, d_input + R*M)
        ctrl_output, ctrl_state = self.controller(
            controller_input, ctrl_state
        )  # (B, controller_size)

        # --- Interface Vector ---
        interface = self.fc_interface(ctrl_output)  # (B, total_param_size)
        interface = torch.clamp(interface, -self.clip_value, self.clip_value)

        # --- Parse Read Parameters ---
        read_params = interface[:, : R * self._read_param_size].reshape(
            -1, R, self._read_param_size
        )  # (B, R, rps)
        read_keys = torch.tanh(read_params[:, :, :M])  # (B, R, M)
        read_strengths = F.softplus(read_params[:, :, M])  # (B, R)
        free_gates = torch.sigmoid(read_params[:, :, M + 1])  # (B, R)
        read_modes = F.softmax(
            read_params[:, :, M + 2 : M + 2 + self.num_read_modes], dim=-1
        )  # (B, R, 1+2W)

        # --- Parse Write Parameters ---
        write_params = interface[:, R * self._read_param_size :].reshape(
            -1, W, self._write_param_size
        )  # (B, W, wps)
        write_keys = torch.tanh(write_params[:, :, :M])  # (B, W, M)
        write_strengths = F.softplus(write_params[:, :, M])  # (B, W)
        erase_vecs = torch.sigmoid(write_params[:, :, M + 1 : 2 * M + 1])  # (B, W, M)
        write_vecs = torch.tanh(write_params[:, :, 2 * M + 1 : 3 * M + 1])  # (B, W, M)
        alloc_gates = torch.sigmoid(write_params[:, :, 3 * M + 1])  # (B, W)
        write_gates = torch.sigmoid(write_params[:, :, 3 * M + 2])  # (B, W)

        # --- Update Usage (from previous read/write weights) ---
        prev_write_w = torch.stack(write_weights, dim=1)  # (B, W, N)
        write_factor = 1 - torch.prod(1 - prev_write_w, dim=1)  # (B, N)
        usage_after_write = usage + (1 - usage) * write_factor  # (B, N)
        prev_read_w = torch.stack(read_weights, dim=1)  # (B, R, N)
        retention = torch.prod(
            1 - free_gates.unsqueeze(-1) * prev_read_w, dim=1
        )  # (B, N)
        usage = usage_after_write * retention  # (B, N)

        # --- Allocation & Write Weights ---
        new_write_weights = []
        alloc_usage = usage
        for i in range(W):
            alloc_w = self._allocation(alloc_usage)  # (B, N)
            wc = self._content_address(
                memory, write_keys[:, i], write_strengths[:, i]
            )  # (B, N)
            ag, wg = (
                alloc_gates[:, i : i + 1],
                write_gates[:, i : i + 1],
            )  # (B, 1), (B, 1)
            w = wg * (ag * alloc_w + (1 - ag) * wc)  # (B, N)
            new_write_weights.append(w)
            alloc_usage = alloc_usage + (1 - alloc_usage) * ag * wg * alloc_w  # (B, N)

        # --- Write to Memory (erase-then-add) ---
        for i in range(W):
            w = new_write_weights[i]  # (B, N)
            e, a = erase_vecs[:, i], write_vecs[:, i]  # (B, M), (B, M)
            erase = torch.bmm(w.unsqueeze(2), e.unsqueeze(1))  # (B, N, M)
            add = torch.bmm(w.unsqueeze(2), a.unsqueeze(1))  # (B, N, M)
            memory = memory * (1 - erase) + add

        # --- Update Link Matrix & Precedence ---
        new_write_w_stacked = torch.stack(new_write_weights, dim=1)  # (B, W, N)
        link_matrix = self._update_link(
            link_matrix, precedence, new_write_w_stacked
        )  # (B, W, N, N)
        precedence = self._update_precedence(
            precedence, new_write_w_stacked
        )  # (B, W, N)

        # --- Read Weights (content + forward/backward from link matrix) ---
        new_read_weights, new_reads = [], []
        for i in range(R):
            rc = self._content_address(
                memory, read_keys[:, i], read_strengths[:, i]
            )  # (B, N)
            prev_rw = read_weights[i].unsqueeze(2)  # (B, N, 1)

            # forward/backward via link matrix
            # link_matrix: (B, W, N, N)
            # prev_rw: (B, N, 1) -> (B, 1, N, 1)
            prev_rw_exp = prev_rw.unsqueeze(1)
            forward_w = torch.matmul(link_matrix, prev_rw_exp).squeeze(3)  # (B, W, N)
            backward_w = torch.matmul(link_matrix.transpose(2, 3), prev_rw_exp).squeeze(
                3
            )  # (B, W, N)

            modes = read_modes[:, i]  # (B, 1+2W)
            backward_mode = modes[:, :W]  # (B, W)
            forward_mode = modes[:, W : 2 * W]  # (B, W)
            content_mode = modes[:, 2 * W :]  # (B, 1)

            w = (
                content_mode * rc
                + (backward_mode.unsqueeze(2) * backward_w).sum(dim=1)
                + (forward_mode.unsqueeze(2) * forward_w).sum(dim=1)
            )  # (B, N)

            new_read_weights.append(w)
            new_reads.append(torch.bmm(w.unsqueeze(1), memory).squeeze(1))  # (B, M)

        # --- Output ---
        output = self.fc_output(
            torch.cat([ctrl_output] + new_reads, dim=1)
        )  # (B, d_model)
        output = torch.clamp(output, -self.clip_value, self.clip_value)
        output = self.dropout_layer(output)

        # --- Pack State ---
        new_state = (
            *ctrl_state,
            memory,
            *new_reads,
            *new_read_weights,
            *new_write_weights,
            link_matrix,
            precedence,
            usage,
        )
        return output, new_state
