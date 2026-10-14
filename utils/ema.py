import copy

import torch


class ModelEMA:
    def __init__(self, model, beta: float, update_every: int, update_after_step: int):
        self.model = model
        self.beta = beta
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.step = 0
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self):
        self.step += 1
        if self.step < self.update_after_step:
            return
        if self.step == self.update_after_step:
            self.copy_from_model()
            return
        if self.step % self.update_every != 0:
            return

        model_state = self.model.state_dict()
        ema_state = self.ema_model.state_dict()
        with torch.no_grad():
            for name, ema_value in ema_state.items():
                model_value = model_state[name].detach().to(device=ema_value.device)
                if ema_value.dtype.is_floating_point:
                    ema_value.lerp_(model_value.to(dtype=ema_value.dtype), 1.0 - self.beta)
                else:
                    ema_value.copy_(model_value)

    def copy_from_model(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def state_dict(self):
        return {"ema_model": self.ema_model.state_dict(), "step": self.step}

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict["ema_model"])
        self.step = int(state_dict["step"])
