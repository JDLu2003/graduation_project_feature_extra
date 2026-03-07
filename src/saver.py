from pathlib import Path
import torch

def save_features(
    features: torch.Tensor,
    output_path: Path,
    skip_existing: bool = True
) -> bool:
    """
    Saves a feature tensor to a specified path.

    Args:
        features: The torch.Tensor to save.
        output_path: The Path where the tensor should be saved (e.g., C_18_U_1.pt).
        skip_existing: If True, skips saving if the file already exists.

    Returns:
        True if features were saved or skipped, False if an error occurred.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and output_path.exists():
        # print(f"Skipping existing feature file: {output_path}")
        return True

    try:
        torch.save(features, output_path) # type: ignore
        # Optional: Load back and verify (as per CLAUDE.md)
        # loaded_features = torch.load(output_path)
        # assert torch.equal(features, loaded_features), "Feature save/load verification failed!"
        return True
    except Exception as e:
        print(f"Error saving features to {output_path}: {e}")
        return False

# Example usage (for testing purposes)
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directory to path to import src modules (for config)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config import AppConfig

    print("--- Testing Feature Saver ---")

    # Create a dummy config (or load real one)
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)

    try:
        app_config = AppConfig.from_yaml(config_path)
        feat_out_dir = app_config.paths.feat_out

        dummy_features = torch.randn(3, 1024) # Example features for 3 persons
        dummy_output_path = feat_out_dir / "Video_en_dev" / "C_999" / "C_999_U_0.pt"

        print(f"Attempting to save dummy features to: {dummy_output_path}")
        success = save_features(dummy_features, dummy_output_path, skip_existing=False)

        if success:
            print("Dummy features saved successfully!")
            # Verify by loading
            loaded_features = torch.load(dummy_output_path) # type: ignore
            assert torch.equal(dummy_features, loaded_features), "Saved features mismatch loaded features!"
            print(f"Loaded features shape: {loaded_features.shape}")

            # Test skip_existing
            print("Attempting to save dummy features again (skip_existing=True)...")
            success_skip = save_features(dummy_features, dummy_output_path, skip_existing=True)
            if success_skip:
                print("Skipped existing features successfully.")
            else:
                print("Failed to skip existing features.")

            # Clean up dummy file
            dummy_output_path.unlink()
            dummy_output_path.parent.rmdir()
            print("Cleaned up dummy files.")

            print("Feature saver test successful!")
        else:
            print("Failed to save dummy features.")
            sys.exit(1)

    except AssertionError as e:
        print(f"Test failed due to assertion error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during test: {e}")
        sys.exit(1)