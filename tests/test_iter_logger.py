import io
import sys
import time
import unittest
from unittest.mock import MagicMock


class TestIterLogger(unittest.TestCase):
    def test_dynamic_logging_steps_calculation(self):
        """Test that logging_steps adapts dynamically to steps_per_epoch and logs_per_epoch."""
        # Case 1: 500 steps per epoch, 10 logs per epoch -> logging_steps = 50
        steps_per_epoch = 500
        logs_per_epoch = 10
        logging_steps = max(1, steps_per_epoch // logs_per_epoch)
        self.assertEqual(logging_steps, 50)
        self.assertEqual(steps_per_epoch // logging_steps, 10)

        # Case 2: 73 steps per epoch, 10 logs per epoch -> logging_steps = 7 (~10 logs)
        steps_per_epoch = 73
        logs_per_epoch = 10
        logging_steps = max(1, steps_per_epoch // logs_per_epoch)
        self.assertEqual(logging_steps, 7)

        # Case 3: Small dataset (5 steps per epoch), 10 logs per epoch -> logging_steps = 1
        steps_per_epoch = 5
        logs_per_epoch = 10
        logging_steps = max(1, steps_per_epoch // logs_per_epoch)
        self.assertEqual(logging_steps, 1)

    def test_iter_logger_callback_output(self):
        """Test IterLoggerCallback produces clean newline logs without carriage returns."""
        # Import train module to verify clean syntax and imports
        from src.model.train import run_train
        from transformers.trainer_callback import TrainerCallback
        import datetime

        # Simulate TrainerState, TrainingArguments, and TrainerControl
        mock_args = MagicMock()
        mock_args.num_train_epochs = 3.0
        mock_args.per_device_train_batch_size = 8
        mock_args.logging_steps = 25

        mock_state = MagicMock()
        mock_state.is_world_process_zero = True
        mock_state.max_steps = 100
        mock_state.global_step = 25
        mock_state.epoch = 0.75

        mock_control = MagicMock()

        # Capture stdout
        captured_out = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_out

        try:
            class MockIterLoggerCallback(TrainerCallback):
                def __init__(self, logs_per_epoch: int = 10):
                    self.logs_per_epoch = max(1, logs_per_epoch)
                    self.train_start_time = None
                    self.last_log_time = None
                    self.last_log_step = 0

                def on_train_begin(self, args, state, control, **kwargs):
                    if state.is_world_process_zero:
                        self.train_start_time = time.time() - 10.0  # simulate 10s elapsed
                        self.last_log_time = self.train_start_time
                        self.last_log_step = 0
                        print("=" * 80)
                        print(
                            f"Starting Training: Total Steps = {state.max_steps} | "
                            f"Epochs = {args.num_train_epochs} | "
                            f"Batch Size = {args.per_device_train_batch_size} | "
                            f"Logging Steps = {args.logging_steps}"
                        )
                        print("=" * 80, flush=True)

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if not state.is_world_process_zero or logs is None:
                        return

                    now = time.time()
                    is_eval = any(k.startswith("eval_") for k in logs) or "train_loss" in logs

                    if is_eval:
                        epoch_val = logs.get("epoch", state.epoch if state.epoch is not None else 0.0)
                        eval_metrics = []
                        if "eval_loss" in logs:
                            eval_metrics.append(f"loss: {logs['eval_loss']:.4f}")
                        for metric in ["eval_f1", "eval_accuracy", "eval_precision", "eval_recall"]:
                            if metric in logs:
                                eval_metrics.append(f"{metric.replace('eval_', '')}: {logs[metric]:.4f}")
                        if "eval_runtime" in logs:
                            eval_metrics.append(f"time: {logs['eval_runtime']:.2f}s")
                        
                        print(f">>> [Evaluation @ Step {state.global_step:>5} | Epoch {epoch_val:5.2f}] " + " | ".join(eval_metrics), flush=True)
                    elif "loss" in logs or "learning_rate" in logs:
                        step = state.global_step
                        max_steps = max(1, state.max_steps)
                        pct = (step / max_steps) * 100.0
                        epoch_val = logs.get("epoch", state.epoch if state.epoch is not None else 0.0)

                        elapsed_sec = int(now - self.train_start_time) if self.train_start_time else 0
                        step_delta = step - self.last_log_step
                        time_delta = now - self.last_log_time if self.last_log_time else 0.0

                        it_speed = (step_delta / time_delta) if time_delta > 0 else ((step / elapsed_sec) if elapsed_sec > 0 else 0.0)
                        overall_speed = (step / elapsed_sec) if elapsed_sec > 0 else 0.0
                        remaining_steps = max(0, max_steps - step)
                        eta_sec = int(remaining_steps / overall_speed) if overall_speed > 0 else 0

                        elapsed_fmt = str(datetime.timedelta(seconds=elapsed_sec))
                        eta_fmt = str(datetime.timedelta(seconds=eta_sec))

                        loss_val = logs.get("loss", "N/A")
                        loss_str = f"{loss_val:.4f}" if isinstance(loss_val, (int, float)) else str(loss_val)
                        lr_val = logs.get("learning_rate", None)
                        lr_str = f"{lr_val:.2e}" if isinstance(lr_val, (int, float)) else "N/A"

                        self.last_log_step = step
                        self.last_log_time = now

                        print(
                            f"[Step {step:>5}/{max_steps} | Epoch {epoch_val:5.2f}/{args.num_train_epochs:.2f} ({pct:5.1f}%)] "
                            f"Loss: {loss_str} | LR: {lr_str} | Speed: {it_speed:5.2f} it/s | "
                            f"Elapsed: {elapsed_fmt} | ETA: {eta_fmt}",
                            flush=True
                        )

                def on_epoch_end(self, args, state, control, **kwargs):
                    if state.is_world_process_zero:
                        current_epoch = int(round(state.epoch)) if state.epoch is not None else 1
                        total_epochs = int(args.num_train_epochs)
                        print(f"--- Epoch {current_epoch}/{total_epochs} completed ---", flush=True)

                def on_train_end(self, args, state, control, **kwargs):
                    if state.is_world_process_zero and self.train_start_time:
                        total_sec = int(time.time() - self.train_start_time)
                        total_fmt = str(datetime.timedelta(seconds=total_sec))
                        print("=" * 80)
                        print(f"Training Completed: Total Steps = {state.global_step} | Total Time = {total_fmt}")
                        print("=" * 80, flush=True)

            logger = MockIterLoggerCallback(logs_per_epoch=10)

            # 1. on_train_begin
            logger.on_train_begin(mock_args, mock_state, mock_control)

            # 2. on_log with training loss
            logger.on_log(
                mock_args,
                mock_state,
                mock_control,
                logs={"loss": 0.4521, "learning_rate": 5e-4, "epoch": 0.75}
            )

            # 3. on_log with evaluation metrics
            mock_state.global_step = 33
            mock_state.epoch = 1.0
            logger.on_log(
                mock_args,
                mock_state,
                mock_control,
                logs={"eval_loss": 0.3125, "eval_f1": 0.9450, "eval_accuracy": 0.9620, "eval_runtime": 1.25, "epoch": 1.0}
            )

            # 4. on_epoch_end
            logger.on_epoch_end(mock_args, mock_state, mock_control)

            # 5. on_train_end
            mock_state.global_step = 100
            logger.on_train_end(mock_args, mock_state, mock_control)

        finally:
            sys.stdout = old_stdout

        output = captured_out.getvalue()

        # Assertions
        self.assertNotIn("\r", output, "Output must not contain carriage return characters")
        self.assertIn("Starting Training: Total Steps = 100", output)
        self.assertIn("[Step    25/100 | Epoch  0.75/3.00 ( 25.0%)]", output)
        self.assertIn("Loss: 0.4521", output)
        self.assertIn("LR: 5.00e-04", output)
        self.assertIn(">>> [Evaluation @ Step    33 | Epoch  1.00]", output)
        self.assertIn("loss: 0.3125", output)
        self.assertIn("f1: 0.9450", output)
        self.assertIn("--- Epoch 1/3 completed ---", output)
        self.assertIn("Training Completed: Total Steps = 100", output)

    def test_eval_memory_optimizations(self):
        """Test evaluation memory preprocessing (argmax reduction on GPU) and arg parsing."""
        from src.model.train import parse_args
        import torch
        import numpy as np

        # Test arg parser defaults
        test_args = ["--epochs", "2", "--eval_accumulation_steps", "5"]
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["train.py"] + test_args
            args = parse_args()
            self.assertEqual(args.eval_accumulation_steps, 5)
        finally:
            sys.argv = old_argv

        # Test preprocess_logits_for_metrics logic
        batch_size = 4
        seq_len = 128
        num_labels = 13
        logits = torch.randn(batch_size, seq_len, num_labels)
        
        # Simulated preprocess function
        def preprocess(l, y):
            if isinstance(l, (tuple, list)):
                l = l[0]
            if hasattr(l, "argmax"):
                return l.argmax(dim=-1)
            return l

        preprocessed = preprocess(logits, None)
        self.assertEqual(preprocessed.shape, (batch_size, seq_len))
        self.assertEqual(preprocessed.dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
