"""
Installation Test Script

Run this script to verify that all dependencies are correctly installed.
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    print("-" * 50)
    
    packages = [
        ('torch', 'PyTorch'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('flask', 'Flask'),
    ]
    
    failed = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name:20s} - OK")
        except ImportError as e:
            print(f"❌ {name:20s} - FAILED: {e}")
            failed.append(name)
    
    print("-" * 50)
    
    if failed:
        print(f"\n❌ {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"   - {pkg}")
        print("\nPlease run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_model_file():
    """Test if model file exists"""
    import os
    print("\nTesting model file...")
    print("-" * 50)
    
    model_path = 'best.pth'
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model file found: {model_path} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        print("   Please ensure best.pth is in the current directory")
        return False

def test_torch_geometric():
    """Test PyTorch Geometric installation"""
    print("\nTesting PyTorch Geometric...")
    print("-" * 50)
    
    try:
        from torch_geometric.nn import GCNConv
        from torch_geometric.data import Data
        import torch
        
        # Create a simple test graph
        x = torch.randn(3, 3)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        
        print(f"✅ PyTorch Geometric is working correctly")
        print(f"   Test graph: {data.num_nodes} nodes, {data.num_edges} edges")
        return True
    except Exception as e:
        print(f"❌ PyTorch Geometric test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe installation"""
    print("\nTesting MediaPipe...")
    print("-" * 50)
    
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        
        # Try to initialize pose
        with mp_pose.Pose(static_image_mode=True) as pose:
            print(f"✅ MediaPipe Pose is working correctly")
            return True
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_opencv():
    """Test OpenCV installation"""
    print("\nTesting OpenCV...")
    print("-" * 50)
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, (255, 0, 0), -1)
        
        print(f"✅ OpenCV is working correctly")
        print(f"   Version: {cv2.__version__}")
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    print("-" * 50)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print(f"ℹ️  CUDA not available (CPU mode will be used)")
            return True
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("Fall Detection System - Installation Test")
    print("="*50 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Model File", test_model_file()))
    results.append(("PyTorch Geometric", test_torch_geometric()))
    results.append(("MediaPipe", test_mediapipe()))
    results.append(("OpenCV", test_opencv()))
    results.append(("CUDA", test_cuda()))
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25s} {status}")
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Run web interface: python web_app.py")
        print("  2. Or test CLI: python fall_detection_test.py <image.jpg>")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
