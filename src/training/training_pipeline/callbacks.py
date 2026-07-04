from transformers import TrainerCallback

class EMACallback(TrainerCallback):
    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.ema_active = False

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        active_model = model.module if hasattr(model, "module") else model
        for name, param in active_model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name] -= (1.0 - self.decay) * (self.shadow[name] - param.data)

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        self._restore_regular_weights(model)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self._apply_ema_weights(model)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._apply_ema_weights(model)
        print("Final EMA weights permanently applied to the model.")

    def _apply_ema_weights(self, model):
        if model is None or self.ema_active:
            return
        active_model = model.module if hasattr(model, "module") else model
        for name, param in active_model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        self.ema_active = True

    def _restore_regular_weights(self, model):
        if model is None or not self.ema_active:
            return
        active_model = model.module if hasattr(model, "module") else model
        for name, param in active_model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()
        self.ema_active = False
