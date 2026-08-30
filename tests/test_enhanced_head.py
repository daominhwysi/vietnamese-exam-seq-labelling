import unittest
import torch
import torch.nn as nn
from transformers import AutoConfig, BertModel, BertConfig
from src.model.head import (
    FocalLoss,
    EnhancedTokenClassificationHead,
    EnhancedTokenClassifierModel,
)

class TestEnhancedHead(unittest.TestCase):
    def test_focal_loss(self):
        batch_size = 2
        seq_len = 8
        num_classes = 13
        
        logits = torch.randn(batch_size, seq_len, num_classes, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, seq_len))
        targets[0, 0] = -100  # Ignored token
        
        loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.05)
        loss = loss_fn(logits, targets)
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)
        
        # Test backward pass
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_enhanced_token_classification_head_multi_layer(self):
        batch_size = 2
        seq_len = 16
        hidden_size = 64
        num_labels = 13
        num_layers = 6
        
        # Simulate hidden states tuple from 6 transformer layers
        hidden_states = tuple(
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        )
        
        head = EnhancedTokenClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            num_layers_to_fuse=4,
            dropout_rates=(0.1, 0.2, 0.3)
        )
        
        # Training mode (Multi-Sample Dropout)
        head.train()
        train_logits = head(hidden_states)
        self.assertEqual(train_logits.shape, (batch_size, seq_len, num_labels))
        
        # Eval mode (Deterministic evaluation)
        head.eval()
        eval_logits = head(hidden_states)
        self.assertEqual(eval_logits.shape, (batch_size, seq_len, num_labels))

    def test_enhanced_token_classification_head_single_tensor(self):
        batch_size = 2
        seq_len = 10
        hidden_size = 64
        num_labels = 7
        
        single_hidden = torch.randn(batch_size, seq_len, hidden_size)
        head = EnhancedTokenClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            num_layers_to_fuse=4
        )
        
        logits = head(single_hidden)
        self.assertEqual(logits.shape, (batch_size, seq_len, num_labels))

    def test_enhanced_token_classifier_model_forward(self):
        config = BertConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            num_labels=13
        )
        base_model = BertModel(config)
        
        model = EnhancedTokenClassifierModel(
            config=config,
            base_model=base_model,
            num_layers_to_fuse=2,
            focal_gamma=2.0
        )
        
        input_ids = torch.randint(0, 100, (2, 8))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, 13, (2, 8))
        
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        self.assertIsNotNone(output.loss)
        self.assertEqual(output.logits.shape, (2, 8, 13))
        self.assertGreater(output.loss.item(), 0.0)

if __name__ == "__main__":
    unittest.main()
