"""
Quick test script to verify your setup is working correctly.
Run this to make sure everything is properly installed and configured.
"""

import sys
import torch
import torchvision
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))


def test_pytorch_installation():
    """Test PyTorch installation and CUDA availability."""
    print("Testing PyTorch Installation...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    # Check device availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available - GPU: {torch.cuda.get_device_name()}")
        device = "cuda"
    elif torch.backends.mps.is_available():
        print("✅ MPS available - Apple Silicon GPU")
        device = "mps"
    else:
        print("✅ CPU only - No GPU detected")
        device = "cpu"

    return device


def test_model_creation():
    """Test ResNet50 model creation."""
    print("\n Testing Model Creation...")

    try:
        from models.resnet import create_resnet50_model

        # Create model
        model = create_resnet50_model()

        # Test with dummy data
        dummy_input = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            outputs = model(dummy_input)
            probabilities = model.predict_proba(dummy_input)

        print(f"✅ Model forward pass successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {outputs.shape}")
        print(f"   Probabilities sum: {probabilities.sum(dim=1)}")

        return True

    except Exception as e:
        print(f" Model test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\n⚙️ Testing Configuration...")

    try:
        from config import DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG

        print(f" Dataset: {DATASET_CONFIG['name']}")
        print(f" Classes: {DATASET_CONFIG['num_classes']}")
        print(f" Models available: {list(MODEL_CONFIG.keys())}")
        print(f" Batch size: {TRAINING_CONFIG['batch_size']}")

        return True

    except Exception as e:
        print(f" Configuration test failed: {e}")
        return False


def test_data_structure():
    """Test data directory structure."""
    print("\n Testing Data Structure...")

    try:
        from config import DATA_DIR, MODELS_DIR, ROOT_DIR

        print(f" Root directory: {ROOT_DIR}")
        print(f" Data directory: {DATA_DIR}")
        print(f" Models directory: {MODELS_DIR}")

        # Check if directories exist
        if DATA_DIR.exists():
            print(f" Data directory exists")
        else:
            print(f"️ Data directory will be created")

        return True

    except Exception as e:
        print(f" Data structure test failed: {e}")
        return False


def test_sample_dataset():
    """Test sample dataset creation."""
    print("\n Testing Sample Dataset...")

    try:
        from data.data_downloader import HAM10000Downloader

        downloader = HAM10000Downloader()
        df = downloader.create_sample_dataset()

        print(f" Sample dataset created with {len(df)} samples")
        print(f" Classes: {df['dx'].unique()}")

        return True

    except Exception as e:
        print(f" Sample dataset test failed: {e}")
        return False


def main():
    """Run all tests."""
    print(" SKIN CANCER DETECTION - SETUP TEST")
    print("=" * 50)

    tests = [
        test_pytorch_installation,
        test_configuration,
        test_data_structure,
        test_model_creation,
        test_sample_dataset
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")

    print("\n" + "=" * 50)
    print(f" RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("ALL TESTS PASSED! Ready to start training!")
    else:
        print("Some tests failed. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    main()