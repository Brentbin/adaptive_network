"""Unit tests for base thinking components."""

import unittest
import torch
from typing import Dict, Any, Tuple

from ...models.enhanced_thinking.base import (
    ThinkingState,
    ThinkingResult,
    ThinkingLevel,
    ConfigurationState,
    MemoryInterface
)

class MockThinkingLevel(ThinkingLevel):
    """Mock implementation of ThinkingLevel for testing."""
    
    def __init__(self, level_id: int):
        super().__init__(level_id)
        self.process_called = False
        self.update_called = False
        self.last_performance = None
        
    def process(self, 
                input_data: torch.Tensor,
                context: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        self.process_called = True
        # Simply return input with a fixed confidence
        return input_data, 0.75
        
    def update(self, performance: float) -> None:
        self.update_called = True
        self.last_performance = performance

class TestThinkingState(unittest.TestCase):
    """Test cases for ThinkingState."""
    
    def setUp(self):
        self.state = ThinkingState(
            level=1,
            confidence=0.8,
            attention_weights=torch.tensor([0.5, 0.5]),
            memory_context={"key": "value"},
            configuration="test_config"
        )
    
    def test_state_initialization(self):
        """Test if ThinkingState initializes correctly."""
        self.assertEqual(self.state.level, 1)
        self.assertEqual(self.state.confidence, 0.8)
        self.assertTrue(torch.equal(self.state.attention_weights, 
                                  torch.tensor([0.5, 0.5])))
        self.assertEqual(self.state.memory_context["key"], "value")
        self.assertEqual(self.state.configuration, "test_config")
        
    def test_optional_fields(self):
        """Test if optional fields can be None."""
        state = ThinkingState(level=1, confidence=0.8)
        self.assertIsNone(state.attention_weights)
        self.assertIsNone(state.memory_context)
        self.assertIsNone(state.configuration)

class TestThinkingLevel(unittest.TestCase):
    """Test cases for ThinkingLevel implementation."""
    
    def setUp(self):
        self.level = MockThinkingLevel(1)
        self.input_tensor = torch.tensor([1.0, 2.0, 3.0])
        self.context = {"test": "context"}
        
    def test_level_initialization(self):
        """Test if ThinkingLevel initializes correctly."""
        self.assertEqual(self.level.level_id, 1)
        self.assertIsNone(self.level.state)
        
    def test_process_method(self):
        """Test if process method works correctly."""
        output, confidence = self.level.process(self.input_tensor, self.context)
        self.assertTrue(self.level.process_called)
        self.assertTrue(torch.equal(output, self.input_tensor))
        self.assertEqual(confidence, 0.75)
        
    def test_update_method(self):
        """Test if update method works correctly."""
        self.level.update(0.9)
        self.assertTrue(self.level.update_called)
        self.assertEqual(self.level.last_performance, 0.9)

class TestThinkingResult(unittest.TestCase):
    """Test cases for ThinkingResult."""
    
    def setUp(self):
        self.output = torch.tensor([1.0, 2.0, 3.0])
        self.state = ThinkingState(level=1, confidence=0.8)
        self.thinking_path = [
            ThinkingState(level=0, confidence=0.6),
            ThinkingState(level=1, confidence=0.8)
        ]
        self.metrics = {"accuracy": 0.9}
        
        self.result = ThinkingResult(
            output=self.output,
            state=self.state,
            thinking_path=self.thinking_path,
            metrics=self.metrics
        )
    
    def test_result_initialization(self):
        """Test if ThinkingResult initializes correctly."""
        self.assertTrue(torch.equal(self.result.output, self.output))
        self.assertEqual(self.result.state, self.state)
        self.assertEqual(len(self.result.thinking_path), 2)
        self.assertEqual(self.result.metrics["accuracy"], 0.9)
        
    def test_thinking_path_integrity(self):
        """Test if thinking path maintains correct sequence."""
        self.assertEqual(self.result.thinking_path[0].level, 0)
        self.assertEqual(self.result.thinking_path[1].level, 1)
        self.assertEqual(self.result.thinking_path[0].confidence, 0.6)
        self.assertEqual(self.result.thinking_path[1].confidence, 0.8)

if __name__ == '__main__':
    unittest.main() 