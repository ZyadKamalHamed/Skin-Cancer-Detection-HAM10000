"""
Test our setup to make sure everything works
"""

def test_pytorch():
    """Test PyTorch installation"""
    try:
        import torch
        import torchvision
        print(" PyTorch Test Results:")
        print(f" PyTorch version: {torch.__version__}")
        print(f" Torchvision version: {torchvision.__version__}")
        print(f" CUDA available: {torch.cuda.is_available()}")

        # Test tensor creation
        x = torch.randn(2, 3)
        print(f" Test tensor shape: {x.shape}")

        # Test device detection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Using device: {device}")

        return True
    except ImportError as e:
        print(f" PyTorch import error: {e}")
        return False

def test_config():
    """Test our configuration"""
    try:
        from config import DATASET_CONFIG, TRAINING_CONFIG
        print("\n Configuration Test Results:")
        print(f" Number of classes: {DATASET_CONFIG['num_classes']}")
        print(f" Image size: {DATASET_CONFIG['image_size']}")
        print(f" Batch size: {DATASET_CONFIG['batch_size']}")
        return True
    except ImportError as e:
        print(f" Config import error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running Setup Tests...\n")

    pytorch_ok = test_pytorch()
    config_ok = test_config()

    if pytorch_ok and config_ok:
        print("\n All tests passed! Ready to build the skin cancer detection system!")
    else:
        print("\n Some tests failed. Let's fix these issues first.")