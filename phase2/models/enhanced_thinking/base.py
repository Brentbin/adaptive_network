"""Base classes and interfaces for the enhanced thinking system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch

@dataclass
class ThinkingState:
    """Represents the state of thinking process."""
    level: int
    confidence: float
    attention_weights: Optional[torch.Tensor] = None
    memory_context: Optional[Dict[str, Any]] = None
    configuration: Optional[str] = None

@dataclass
class ThinkingResult:
    """Represents the result of a thinking process."""
    output: torch.Tensor
    state: ThinkingState
    thinking_path: List[ThinkingState]
    metrics: Dict[str, float]

class ThinkingLevel(ABC):
    """Abstract base class for a thinking level."""
    
    def __init__(self, level_id: int):
        self.level_id = level_id
        self.state: Optional[ThinkingState] = None
    
    @abstractmethod
    def process(self, 
                input_data: torch.Tensor,
                context: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """Process input data at this thinking level.
        
        Args:
            input_data: Input tensor to process
            context: Additional context information
            
        Returns:
            Tuple of (processed_output, confidence)
        """
        pass
    
    @abstractmethod
    def update(self, performance: float) -> None:
        """Update level based on performance feedback."""
        pass

class ConfigurationState(ABC):
    """Abstract base class for configuration states."""
    
    @abstractmethod
    def apply(self, 
              network: Any,
              context: Dict[str, Any]) -> None:
        """Apply this configuration state to the network."""
        pass
    
    @abstractmethod
    def evaluate(self, 
                 performance: float,
                 context: Dict[str, Any]) -> float:
        """Evaluate the effectiveness of this configuration."""
        pass

class MemoryInterface(ABC):
    """Abstract base class for memory management."""
    
    @abstractmethod
    def store(self,
              key: str,
              value: Any,
              context: Dict[str, Any]) -> None:
        """Store information in memory."""
        pass
    
    @abstractmethod
    def retrieve(self,
                key: str,
                context: Dict[str, Any]) -> Any:
        """Retrieve information from memory."""
        pass
    
    @abstractmethod
    def update(self,
               key: str,
               value: Any,
               context: Dict[str, Any]) -> None:
        """Update existing memory."""
        pass 