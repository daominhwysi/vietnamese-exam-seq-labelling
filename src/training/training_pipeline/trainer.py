import torch
from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, real_upsample_factor=1.0, use_focal_loss=False, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.real_upsample_factor = real_upsample_factor
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

    def get_train_dataloader(self):
        """Override to inject WeightedRandomSampler for real vs synthetic upsampling."""
        if self.real_upsample_factor <= 1.0:
            return super().get_train_dataloader()

        train_dataset = self.train_dataset
        n_real = 0
        n_synth = 0
        sample_weights = []

        for sample in train_dataset:
            meta = sample.get("metadata", {})
            is_real = meta.get("is_real", False) if isinstance(meta, dict) else False
            if is_real:
                sample_weights.append(self.real_upsample_factor)
                n_real += 1
            else:
                sample_weights.append(1.0)
                n_synth += 1

        print(
            f"[WeightedSampler] Synth samples: {n_synth}, Real samples: {n_real} "
            f"(effective weight: synth=1.0, real={self.real_upsample_factor})"
        )

        if n_real == 0:
            print("[WeightedSampler] Warning: no real samples found in train split (is_real=False for all). "
                  "Falling back to uniform sampling. Re-run prepare-dataset to propagate is_real metadata.")
            return super().get_train_dataloader()

        from torch.utils.data import WeightedRandomSampler, DataLoader

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(self.args.seed),
        )

        train_dataset = self._remove_unused_columns(train_dataset, description="training")

        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _compute_focal_loss(self, logits, labels):
        num_labels = self.model.config.num_labels
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        valid_mask = labels_flat != -100
        logits_valid = logits_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]

        if logits_valid.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_probs = torch.nn.functional.log_softmax(logits_valid, dim=-1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(dim=-1, index=labels_valid.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(dim=-1, index=labels_valid.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - target_probs) ** self.focal_gamma

        if self.class_weights is not None:
            alpha_weight = self.class_weights[labels_valid]
            focal_weight = focal_weight * alpha_weight

        loss = -focal_weight * target_log_probs
        return loss.mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        if getattr(self.args, "past_index", -1) >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            logits = outputs.get("logits")
            if self.use_focal_loss:
                loss = self._compute_focal_loss(logits, labels)
            elif self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            else:
                loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        else:
            loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

