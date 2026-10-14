import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.cells.base import RecurrentCell
from models.cells.lstm import LSTMCell
from models.cells.rnn import RNNCell


_CONTROLLERS = {"LSTM": LSTMCell, "RNN": RNNCell}


class NAMTuringMachineCell(RecurrentCell):
    def __init__(self, d_input: int, d_model: int, controller_type: str = "LSTM",
                 num_tapes: int = 4, tape_length: int = 32, head_dim: int = 64,
                 dropout: float = 0.0, clip_value: float = 20.0):
        super().__init__(d_input=d_input, d_model=d_model)
        self.num_tapes, self.tape_length, self.head_dim = num_tapes, tape_length, head_dim
        self.clip_value = clip_value

        # --- Controller ---
        ctrl_cls = _CONTROLLERS[controller_type]
        self.controller = ctrl_cls(d_input=d_input + num_tapes * head_dim, d_model=d_model, dropout=dropout)
        self._ctrl_state_size = len(self.controller.initialize_state())

        # --- Value and Key layers (project from RAW INPUT) ---
        self.value_layer = nn.Linear(d_input, num_tapes * head_dim)
        self.key_layer = nn.Linear(d_input, num_tapes * head_dim)

        # --- Action and Query layers (project from CONTROLLER OUTPUT) ---
        # Per-tape actions: 4 read_dirs + 4 write_dirs + 3 rwe_probs = 11
        self.action_layer = nn.Linear(d_model, 11 * num_tapes)
        self.query_layer = nn.Linear(d_model, num_tapes * head_dim)

        # --- Output Projection ---
        self.fc_output = nn.Linear(d_model + num_tapes * head_dim, d_model)

        # --- Device anchor for initialize_state ---
        self.register_buffer("_device_anchor", torch.zeros(1))

        # --- State unpacking indices ---
        self._idx_tape = self._ctrl_state_size
        self._idx_tape_key = self._ctrl_state_size + 1
        self._idx_rpos = self._ctrl_state_size + 2
        self._idx_wpos = self._ctrl_state_size + 3
        self._idx_read_vec = self._ctrl_state_size + 4

        self.dropout_layer = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.value_layer, self.key_layer, self.action_layer, self.query_layer):
            nn.init.xavier_uniform_(layer.weight, gain=1.4)
            nn.init.normal_(layer.bias, std=0.01)
        nn.init.xavier_uniform_(self.fc_output.weight, gain=1)
        nn.init.normal_(self.fc_output.bias, std=0.01)

    def initialize_state(self):
        dev = self._device_anchor.device
        ctrl_state = self.controller.initialize_state()  # tuple of (1, d_model) tensors
        tape = torch.zeros(1, self.tape_length, self.num_tapes, self.head_dim, device=dev)  # (1, L, T, D)
        tape_key = torch.zeros(1, self.tape_length, self.num_tapes, self.head_dim, device=dev)  # (1, L, T, D)
        rpos = torch.zeros(1, self.tape_length, self.num_tapes, device=dev)  # (1, L, T)
        rpos[:, 0, :] = 1.0
        wpos = torch.zeros(1, self.tape_length, self.num_tapes, device=dev)  # (1, L, T)
        wpos[:, 0, :] = 1.0
        read_vec = torch.zeros(1, self.num_tapes * self.head_dim, device=dev)  # (1, T*D)
        return (*ctrl_state, tape, tape_key, rpos, wpos, read_vec)

    def forward(self, input: Tensor, state):
        assert input.ndim == 2, "Input should be of size B x D"
        B = input.size(0)
        T, D = self.num_tapes, self.head_dim

        # --- Unpack State ---
        ctrl_state = state[:self._ctrl_state_size]
        tape = state[self._idx_tape]  # (B, L, T, D)
        tape_key = state[self._idx_tape_key]  # (B, L, T, D)
        rpos = state[self._idx_rpos]  # (B, L, T)
        wpos = state[self._idx_wpos]  # (B, L, T)
        read_vec = state[self._idx_read_vec]  # (B, T*D)

        # --- Controller ---
        ctrl_input = torch.cat([input, read_vec], dim=1)  # (B, d_input + T*D)
        ctrl_output, ctrl_state = self.controller(ctrl_input, ctrl_state)  # (B, d_model)

        # --- Values and Keys from raw input ---
        values = self.value_layer(input).view(B, T, D)  # (B, T, D)
        keys = F.normalize(self.key_layer(input).view(B, T, D), dim=-1)  # (B, T, D)

        # --- Actions and Queries from controller output ---
        actions = self.action_layer(ctrl_output).view(B, T, 11)  # (B, T, 11)
        actions = torch.clamp(actions, -self.clip_value, self.clip_value)
        read_dirs = F.softmax(actions[:, :, :4], dim=-1)  # (B, T, 4) — left, stay, right, jump
        write_dirs = F.softmax(actions[:, :, 4:8], dim=-1)  # (B, T, 4)
        rwe = torch.sigmoid(actions[:, :, 8:])  # (B, T, 3) — read, write, erase probs

        queries = F.normalize(self.query_layer(ctrl_output).view(B, T, D), dim=-1)  # (B, T, D)

        # --- Normalize positions for tape read/write only ---
        nwpos = F.normalize(wpos, dim=1)  # (B, L, T)
        nrpos = F.normalize(rpos, dim=1)  # (B, L, T)

        # --- Write to tape ---
        old_val = torch.einsum('bltd,blt->btd', tape, nwpos)  # (B, T, D)
        delta = values * rwe[:, :, 1:2] - old_val * rwe[:, :, 2:3]  # (B, T, D)
        tape = tape + torch.einsum('blt,btd->bltd', nwpos, delta)  # (B, L, T, D)

        # --- Read from tape ---
        read_out = torch.einsum('bltd,blt->btd', tape, nrpos) * rwe[:, :, 0:1]  # (B, T, D)

        # --- Update key tape (using raw wpos) ---
        old_key_proj = torch.einsum('bltd,btd->blt', tape_key, keys)  # (B, L, T)
        pw_2d = rwe[:, :, 1].unsqueeze(1)  # (B, 1, T)
        pe_2d = rwe[:, :, 2].unsqueeze(1)  # (B, 1, T)
        key_weight = wpos * pw_2d - old_key_proj * pe_2d  # (B, L, T)
        tape_key = tape_key + torch.einsum('blt,btd->bltd', key_weight, keys)  # (B, L, T, D)

        # --- Jump position via key tape ---
        jpos = torch.einsum('bltd,btd->blt', tape_key, queries)  # (B, L, T)

        # --- Update write position (using raw wpos) ---
        prev_w = torch.roll(wpos, -1, dims=1)  # (B, L, T) — left
        next_w = torch.roll(wpos, 1, dims=1)  # (B, L, T) — right
        wpos = (prev_w * write_dirs[:, :, 0].unsqueeze(1) +  # left
                wpos * write_dirs[:, :, 1].unsqueeze(1) +  # stay
                next_w * write_dirs[:, :, 2].unsqueeze(1) +  # right
                jpos * write_dirs[:, :, 3].unsqueeze(1))  # jump

        # --- Update read position (using raw rpos) ---
        prev_r = torch.roll(rpos, -1, dims=1)  # (B, L, T) — left
        next_r = torch.roll(rpos, 1, dims=1)  # (B, L, T) — right
        rpos = (prev_r * read_dirs[:, :, 0].unsqueeze(1) +  # left
                rpos * read_dirs[:, :, 1].unsqueeze(1) +  # stay
                next_r * read_dirs[:, :, 2].unsqueeze(1) +  # right
                jpos * read_dirs[:, :, 3].unsqueeze(1))  # jump

        # --- Output ---
        read_flat = read_out.reshape(B, -1)  # (B, T*D)
        output = self.fc_output(torch.cat([ctrl_output, read_flat], dim=1))  # (B, d_model)
        output = torch.clamp(output, -self.clip_value, self.clip_value)
        output = self.dropout_layer(output)

        # --- Pack State ---
        new_state = (*ctrl_state, tape, tape_key, rpos, wpos, read_flat)
        return output, new_state
