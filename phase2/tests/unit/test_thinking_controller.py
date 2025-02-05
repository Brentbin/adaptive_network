"""Unit tests for enhanced thinking controller."""

import unittest
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

from ...models.enhanced_thinking.thinking_controller import (
    EnhancedThinkingController,
    ControllerConfig
)
from ...models.enhanced_thinking.base import ThinkingLevel

class SimpleThinkingLevel(ThinkingLevel):
    """Simple implementation of ThinkingLevel for testing."""
    
    def __init__(self, level_id: int):
        super().__init__(level_id)
        self.linear = nn.Linear(3, 3)  # Simple linear transformation
        
    def process(self, 
                input_data: torch.Tensor,
                context: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        output = self.linear(input_data)
        # Compute confidence based on output norm
        confidence = torch.norm(output) / torch.norm(input_data)
        confidence = min(confidence.item(), 1.0)
        return output, confidence
        
    def update(self, performance: float) -> None:
        # Simple update - no actual implementation needed for tests
        pass

class TestEnhancedThinkingController(unittest.TestCase):
    """Test cases for EnhancedThinkingController."""
    
    def setUp(self):
        self.config = ControllerConfig(
            num_levels=3,
            base_confidence_threshold=0.7,
            max_thinking_steps=10,
            attention_heads=4
        )
        
        # Patch the _create_level method
        EnhancedThinkingController._create_level = lambda self, level_id: SimpleThinkingLevel(level_id)
        
        self.controller = EnhancedThinkingController(self.config)
        self.input_tensor = torch.tensor([1.0, 2.0, 3.0])
        self.context = {"test": "context"}
        
    def test_controller_initialization(self):
        """Test if controller initializes correctly."""
        self.assertEqual(len(self.controller.levels), 3)
        self.assertEqual(len(self.controller.confidence_thresholds), 3)
        self.assertEqual(len(self.controller.thinking_path), 0)
        
    def test_forward_pass(self):
        """Test if forward pass works correctly."""
        result = self.controller(self.input_tensor, self.context)
        
        # Check result structure
        self.assertIsNotNone(result.output)
        self.assertIsNotNone(result.state)
        self.assertGreater(len(result.thinking_path), 0)
        self.assertIsNotNone(result.metrics)
        
        # Check metrics
        self.assertIn('num_steps', result.metrics)
        self.assertIn('final_confidence', result.metrics)
        self.assertIn('avg_confidence', result.metrics)
        
    def test_thinking_path_generation(self):
        """Test if thinking path is generated correctly."""
        result = self.controller(self.input_tensor, self.context)
        
        # Check thinking path properties
        self.assertGreater(len(result.thinking_path), 0)
        self.assertLessEqual(len(result.thinking_path), self.config.num_levels)
        
        # Check level sequence
        for i, state in enumerate(result.thinking_path):
            self.assertEqual(state.level, i)
            
    def test_confidence_based_stopping(self):
        """Test if controller stops when confidence threshold is met."""
        # Set very low confidence threshold
        self.controller.confidence_thresholds.data.fill_(0.1)
        result = self.controller(self.input_tensor, self.context)
        
        # Should stop after first level
        self.assertEqual(len(result.thinking_path), 1)
        
        # Set very high confidence threshold
        self.controller.confidence_thresholds.data.fill_(0.99)
        result = self.controller(self.input_tensor, self.context)
        
        # Should use all levels
        self.assertEqual(len(result.thinking_path), self.config.num_levels)
        
    def test_update_mechanism(self):
        """Test if update mechanism works correctly."""
        # First do a forward pass
        result = self.controller(self.input_tensor, self.context)
        
        # Then update with good performance
        initial_thresholds = self.controller.confidence_thresholds.clone()
        self.controller.update(0.9)  # Good performance
        
        # Thresholds should increase
        self.assertTrue(torch.all(self.controller.confidence_thresholds >= initial_thresholds))
        
        # Update with poor performance
        self.controller.confidence_thresholds = initial_thresholds.clone()
        self.controller.update(0.3)  # Poor performance
        
        # Thresholds should decrease
        self.assertTrue(torch.all(self.controller.confidence_thresholds <= initial_thresholds))
        
    def test_metrics_computation(self):
        """Test if metrics are computed correctly."""
        result = self.controller(self.input_tensor, self.context)
        
        # Check metrics structure
        self.assertIsInstance(result.metrics, dict)
        self.assertIn('num_steps', result.metrics)
        self.assertIn('final_confidence', result.metrics)
        self.assertIn('avg_confidence', result.metrics)
        
        # Check metrics values
        self.assertEqual(result.metrics['num_steps'], len(result.thinking_path))
        self.assertEqual(result.metrics['final_confidence'], 
                        result.thinking_path[-1].confidence)
        
        # Check average confidence calculation
        expected_avg = sum(s.confidence for s in result.thinking_path) / len(result.thinking_path)
        self.assertAlmostEqual(result.metrics['avg_confidence'], expected_avg)

if __name__ == '__main__':
    unittest.main() 