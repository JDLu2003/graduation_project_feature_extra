import torch
import torch.nn as nn
from typing import Literal

class LinearProjection(nn.Module):
    """
    A simple linear projection layer to transform feature dimensions.
    Used to project CLIP's output dimension (e.g., 512) to the target dimension (e.g., 1024).
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the linear projection.

        Args:
            x: Input tensor with shape (..., input_dim).

        Returns:
            Output tensor with shape (..., output_dim).
        """
        return self.projection(x)

# Example usage (for testing purposes)
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import AppConfig

    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)

    try:
        app_config = AppConfig.from_yaml(config_path)
        # Assuming visual_clip_config is active and exists
        clip_config = app_config.extractor.visual_clip_config
        input_dim = clip_config.clip_output_dim
        target_dim = clip_config.target_dim

        print(f"--- Testing LinearProjection ---")
        print(f"Input Dimension: {input_dim}, Target Dimension: {target_dim}")

        # Create a dummy input tensor
        dummy_input = torch.randn(1, input_dim) # Batch size 1
        print(f"Dummy input shape: {dummy_input.shape}")

        # Instantiate and test the projection layer
        projection_layer = LinearProjection(input_dim, target_dim)
        output = projection_layer(dummy_input)

        assert output.shape == (1, target_dim), \
            f"Expected output shape (1, {target_dim}), but got {output.shape}"

        print(f"Projection output shape: {output.shape}")
        print("LinearProjection test successful!")

    except AssertionError as e:
        print(f"Test failed due to assertion error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during test: {e}")
        sys.exit(1)