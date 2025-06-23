"""
ResNet50 Architecture for Skin Cancer Detection

This module implements a ResNet50 model with transfer learning specifically
optimized for medical image classification tasks.

Key Concepts:
- Transfer Learning: Using ImageNet pre-trained weights as starting point
- Feature Extraction: Lower layers detect edges, textures (universal features)
- Fine-tuning: Higher layers adapt to medical domain specifics
- Dropout: Prevents overfitting on limited medical data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict


class SkinCancerResNet50(nn.Module):
    """
    ResNet50 model adapted for skin cancer detection.

    Architecture:
    1. ResNet50 backbone (pre-trained on ImageNet)
    2. Global Average Pooling (reduces overfitting vs fully connected)
    3. Dropout layer (prevents overfitting)
    4. Classification head (7 skin cancer classes)

    Why ResNet50?
    - Proven performance on medical images
    - Skip connections help with gradient flow
    - 25M parameters - manageable for limited data
    - Strong ImageNet features transfer well to skin images
    """

    def __init__(
            self,
            num_classes: int = 7,
            pretrained: bool = True,
            dropout_rate: float = 0.5,
            freeze_backbone: bool = False
    ):
        """
        Initialize the ResNet50 model.

        Args:
            num_classes: Number of skin cancer classes (7 for HAM10000)
            pretrained: Use ImageNet pre-trained weights
            dropout_rate: Dropout probability for regularization
            freeze_backbone: Whether to freeze backbone weights
        """
        super(SkinCancerResNet50, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Load pre-trained ResNet50
        # This gives us millions of learned features from natural images
        self.backbone = models.resnet50(pretrained=pretrained)

        # Remove the original classification layer
        # ResNet50 ends with AdaptiveAvgPool2d + Linear(2048, 1000)
        # We'll replace this with our medical-specific head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Feature dimension from ResNet50 conv layers
        self.feature_dim = 2048

        # Global Average Pooling
        # Converts (batch, 2048, 7, 7) -> (batch, 2048)
        # More robust than flattening for variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout for regularization
        # Critical for medical imaging with limited data
        self.dropout = nn.Dropout(dropout_rate)

        # Classification head
        # Maps from 2048 features to 7 skin cancer classes
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # Initialize classifier weights
        # Xavier initialization works well for medical classification
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

        # Optionally freeze backbone for pure feature extraction
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze backbone weights for feature extraction only."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen - only training classification head")

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔓 Backbone unfrozen - fine-tuning entire network")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, 3, 224, 224)

        Returns:
            logits: Raw predictions (batch_size, num_classes)
        """
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.dim()}D")

        if x.size(1) != 3:
            raise ValueError(f"Expected 3 input channels (RGB), got {x.size(1)}")

        # Extract features using ResNet50 backbone
        # x: (batch, 3, 224, 224) -> (batch, 2048, 7, 7)
        features = self.backbone(x)

        # Global average pooling
        # (batch, 2048, 7, 7) -> (batch, 2048, 1, 1)
        pooled = self.global_pool(features)

        # Flatten to vector
        # (batch, 2048, 1, 1) -> (batch, 2048)
        flattened = torch.flatten(pooled, 1)

        # Apply dropout for regularization
        dropped = self.dropout(flattened)

        # Final classification
        # (batch, 2048) -> (batch, 7)
        logits = self.classifier(dropped)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations before classification.
        Useful for visualization and analysis.

        Args:
            x: Input tensor (batch_size, 3, 224, 224)

        Returns:
            features: Feature vector (batch_size, 2048)
        """
        with torch.no_grad():
            features = self.backbone(x)
            pooled = self.global_pool(features)
            flattened = torch.flatten(pooled, 1)
        return flattened

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities using softmax.

        Args:
            x: Input tensor (batch_size, 3, 224, 224)

        Returns:
            probabilities: Class probabilities (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_model_info(self) -> Dict[str, any]:
        """Get detailed model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": "ResNet50",
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "dropout_rate": self.dropout_rate,
            "feature_dim": self.feature_dim,
            "backbone_frozen": not next(self.backbone.parameters()).requires_grad
        }


def create_resnet50_model(config: Optional[Dict] = None) -> SkinCancerResNet50:
    """
    Factory function to create ResNet50 model from configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        model: Initialized ResNet50 model
    """
    if config is None:
        # Default configuration
        config = {
            "num_classes": 7,
            "pretrained": True,
            "dropout_rate": 0.5,
            "freeze_backbone": False
        }

    model = SkinCancerResNet50(
        num_classes=config.get("num_classes", 7),
        pretrained=config.get("pretrained", True),
        dropout_rate=config.get("dropout_rate", 0.5),
        freeze_backbone=config.get("freeze_backbone", False)
    )

    print(f"✅ ResNet50 model created with {model.get_model_info()['total_parameters']:,} parameters")

    return model


# Example usage and testing
if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_resnet50_model()

    # Create dummy input (batch of 4 images)
    dummy_input = torch.randn(4, 3, 224, 224)

    # Test forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
        probabilities = model.predict_proba(dummy_input)
        features = model.get_features(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Model info: {model.get_model_info()}")

    print("🎉 ResNet50 model test passed!")