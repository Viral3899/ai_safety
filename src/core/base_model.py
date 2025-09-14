"""
Base model architecture for AI Safety Models.

This module provides the foundational classes and interfaces for all safety models
in the AI Safety POC system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum


class SafetyLevel(Enum):
    """Safety risk levels for content classification."""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


@dataclass
class SafetyResult:
    """Result from a safety model prediction."""
    label: str
    score: float
    safety_level: SafetyLevel
    confidence: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelConfig:
    """Configuration class for safety models."""
    model_type: str
    threshold: float = 0.5
    max_length: int = 512
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    device: str = "cpu"
    multilingual: bool = False


class BaseModel(ABC):
    """Abstract base class for all AI Safety Models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        
    @abstractmethod
    def preprocess(self, text: Union[str, List[str]]) -> Any:
        """Preprocess input text for model inference."""
        pass
    
    @abstractmethod
    def predict(self, text: Union[str, List[str]]) -> SafetyResult:
        """Make predictions on preprocessed text."""
        pass
    
    @abstractmethod
    def train(self, train_data: List[Dict[str, Any]], 
              val_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """Train the model on provided data."""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        pass
    
    def get_safety_level(self, confidence: float) -> SafetyLevel:
        """Convert confidence score to safety level."""
        if confidence >= 0.8:
            return SafetyLevel.CRITICAL
        elif confidence >= 0.6:
            return SafetyLevel.HIGH_RISK
        elif confidence >= 0.4:
            return SafetyLevel.MEDIUM_RISK
        elif confidence >= 0.2:
            return SafetyLevel.LOW_RISK
        else:
            return SafetyLevel.SAFE


class SimpleTextCNN(nn.Module):
    """Simple CNN architecture for text classification."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int, 
                 num_filters: int = 100, filter_sizes: List[int] = [3, 4, 5]):
        super(SimpleTextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (filter_size, embedding_dim))
            for filter_size in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))
            pooled = torch.max_pool2d(conv_out, (conv_out.size(2), 1))
            conv_outputs.append(pooled.squeeze(3))
            
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class TransformerBasedModel(nn.Module):
    """Transformer-based model for text classification."""
    
    def __init__(self, model_name: str, num_classes: int, dropout_rate: float = 0.1):
        super(TransformerBasedModel, self).__init__()
        
        from transformers import AutoModel
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)
