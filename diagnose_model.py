"""
Model Diagnosis Script

This script helps diagnose issues with the fall detection model:
1. Check if label mapping is correct (fall=1, not_fall=0 or vice versa)
2. Analyze prediction distribution
3. Test with sample images
"""

import os
import json
import torch
import numpy as np
from fall_detection_test import FallDetector, extract_skeleton_from_image, landmarks_to_graph, make_undirected_edge_index, skeleton_edges

def diagnose_model(model_path='best.pth'):
    """
    Diagnose the model to understand its behavior
    """
    print("="*60)
    print("Fall Detection Model Diagnosis")
    print("="*60)
    
    # Load model
    print("\n1. Loading model...")
    detector = FallDetector(model_path)
    model = detector.model
    device = detector.device
    
    # Check model output range
    print("\n2. Testing model output range...")
    print("   Creating dummy skeleton data...")
    
    # Create test skeletons
    test_cases = {
        "Standing (upright)": create_standing_skeleton(),
        "Falling (horizontal)": create_falling_skeleton(),
        "Random": create_random_skeleton()
    }
    
    print("\n3. Model predictions on synthetic skeletons:")
    print("-" * 60)
    
    for name, landmarks in test_cases.items():
        edge_index = make_undirected_edge_index(skeleton_edges)
        graph = landmarks_to_graph(landmarks, edge_index)
        
        model.eval()
        graph.x = graph.x.to(device)
        graph.edge_index = graph.edge_index.to(device)
        batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
        
        with torch.no_grad():
            out = model(graph.x, graph.edge_index, batch)
            out = out.cpu()
            raw_value = float(out.squeeze().item())
            
            # Current logic
            current_pred = 'fall' if raw_value >= 0.5 else 'not_fall'
            
            # Inverted logic
            inverted_pred = 'not_fall' if raw_value >= 0.5 else 'fall'
            
            print(f"\n{name}:")
            print(f"  Raw output: {raw_value:.4f}")
            print(f"  Current prediction: {current_pred}")
            print(f"  Inverted prediction: {inverted_pred}")
    
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("="*60)
    print("\nIf the model is predicting incorrectly:")
    print("1. Check if 'Inverted prediction' makes more sense")
    print("2. The label mapping might be reversed")
    print("3. Model might have been trained with opposite labels")
    
    print("\nPossible solutions:")
    print("A. Invert the prediction threshold")
    print("B. Retrain the model with correct labels")
    print("C. Add data normalization/preprocessing")
    
    return detector

def create_standing_skeleton():
    """Create a synthetic standing pose"""
    landmarks = {}
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    # Standing pose: head at top (y=0.2), feet at bottom (y=0.9)
    for i, name in enumerate(landmark_names):
        if 'eye' in name or 'nose' in name or 'ear' in name or 'mouth' in name:
            y = 0.2  # Head
        elif 'shoulder' in name:
            y = 0.35
        elif 'elbow' in name:
            y = 0.5
        elif 'wrist' in name or 'hand' in name or 'pinky' in name or 'index' in name or 'thumb' in name:
            y = 0.6
        elif 'hip' in name:
            y = 0.55
        elif 'knee' in name:
            y = 0.7
        else:  # ankle, heel, foot
            y = 0.9
        
        x = 0.5 + (0.1 if 'left' in name else -0.1 if 'right' in name else 0)
        landmarks[name] = {'x': x, 'y': y, 'z': 0.0}
    
    return landmarks

def create_falling_skeleton():
    """Create a synthetic falling pose (horizontal)"""
    landmarks = {}
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    # Falling pose: body horizontal (y around 0.5, x varies)
    for i, name in enumerate(landmark_names):
        if 'eye' in name or 'nose' in name or 'ear' in name or 'mouth' in name:
            x = 0.2  # Head on left
        elif 'shoulder' in name:
            x = 0.35
        elif 'elbow' in name:
            x = 0.5
        elif 'wrist' in name or 'hand' in name or 'pinky' in name or 'index' in name or 'thumb' in name:
            x = 0.6
        elif 'hip' in name:
            x = 0.55
        elif 'knee' in name:
            x = 0.7
        else:  # ankle, heel, foot
            x = 0.85
        
        y = 0.5 + (0.05 if 'left' in name else -0.05 if 'right' in name else 0)
        landmarks[name] = {'x': x, 'y': y, 'z': 0.0}
    
    return landmarks

def create_random_skeleton():
    """Create a random skeleton for testing"""
    landmarks = {}
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    for name in landmark_names:
        landmarks[name] = {
            'x': np.random.uniform(0.3, 0.7),
            'y': np.random.uniform(0.2, 0.8),
            'z': np.random.uniform(-0.5, 0.5)
        }
    
    return landmarks

def test_with_real_images(image_dir=None):
    """
    Test with real images if available
    """
    if image_dir is None or not os.path.exists(image_dir):
        print("\nNo test images directory provided or found.")
        return
    
    print("\n" + "="*60)
    print("Testing with real images")
    print("="*60)
    
    detector = FallDetector('best.pth')
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not image_files:
        print("No images found in directory")
        return
    
    for img_file in image_files[:5]:  # Test first 5 images
        img_path = os.path.join(image_dir, img_file)
        label, conf, _ = detector.predict(img_path)
        print(f"\n{img_file}:")
        print(f"  Prediction: {label}")
        print(f"  Confidence: {conf:.2%}")

if __name__ == '__main__':
    import sys
    
    # Run diagnosis
    detector = diagnose_model('best.pth')
    
    # Test with real images if directory provided
    if len(sys.argv) > 1:
        test_with_real_images(sys.argv[1])
    
    print("\n" + "="*60)
    print("Diagnosis complete!")
    print("="*60)
