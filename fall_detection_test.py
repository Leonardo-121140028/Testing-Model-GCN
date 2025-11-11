"""
Fall Detection Testing Module

This module provides functionality to predict whether a person in an image is falling or not.
It uses MediaPipe for skeleton extraction and a pre-trained GCN model for classification.
"""

import os
import json
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import mediapipe as mp
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
from torch.nn import BatchNorm1d, Dropout, Linear


# ---------------------------
# ====== Model Definition ===
# ---------------------------
skeleton_edges = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15), (15, 17),
    (15, 19), (15, 21), (17, 19), (19, 21),
    (11, 23), (23, 25), (25, 27), (27, 29), (27, 31),
    (23, 24), (24, 26), (26, 28), (28, 30), (28, 32)
]

LANDMARK_NAMES = [
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


class SkeletonGCN(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int | list,
        num_classes: int = 1,
        dropout_rate: float = 0.3,
        pool_type: str = 'mean',
        residual: bool = True,
        seed: int = 42
    ):
        super(SkeletonGCN, self).__init__()
        torch.manual_seed(seed)

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels, hidden_channels]

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.residual = residual

        # input layer
        self.convs.append(GCNConv(num_node_features, hidden_channels[0]))
        self.batch_norms.append(BatchNorm1d(hidden_channels[0]))

        # hidden
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i + 1]))
            self.batch_norms.append(BatchNorm1d(hidden_channels[i + 1]))

        # head
        self.dropout = Dropout(p=dropout_rate)
        self.final_lin1 = Linear(hidden_channels[-1], max(1, hidden_channels[-1] // 2))
        self.final_lin2 = Linear(max(1, hidden_channels[-1] // 2), num_classes)

        # pooling
        if pool_type == 'max':
            self.pool = global_max_pool
        elif pool_type == 'add':
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool

    def forward(self, x, edge_index, batch):
        previous = None
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            if i > 0 and self.residual:
                previous = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            if i > 0 and self.residual and x.size(-1) == previous.size(-1):
                x = x + previous

        x = self.pool(x, batch)
        x = self.dropout(x)
        x = F.relu(self.final_lin1(x))
        x = self.dropout(x)
        x = self.final_lin2(x)

        # If binary (out_features == 1) -> sigmoid, else log_softmax
        if self.final_lin2.out_features == 1:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=-1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        self.final_lin1.reset_parameters()
        self.final_lin2.reset_parameters()


# ---------------------------
# ====== Skeleton Extraction
# ---------------------------
def extract_skeleton_from_image(image_path: str) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Extract skeleton landmarks from an image using MediaPipe Pose.
    
    Args:
        image_path (str): Path to the image file or numpy array
        
    Returns:
        Optional[Dict]: Dictionary of landmarks or None if no pose detected
    """
    mp_pose = mp.solutions.pose
    
    try:
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None
        else:
            image = image_path
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize pose detection
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            # Process image
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract landmarks as dictionary
                landmarks = {}
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmark_name = mp_pose.PoseLandmark(idx).name.lower()
                    landmarks[landmark_name] = {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    }
                return landmarks
            
            return None
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


def draw_skeleton_on_image(image, landmarks: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Draw skeleton on image for visualization.
    
    Args:
        image: Input image (numpy array or path)
        landmarks: Dictionary of skeleton landmarks
        
    Returns:
        np.ndarray: Image with skeleton drawn
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    else:
        image = image.copy()
    
    if image is None:
        return None
    
    height, width = image.shape[:2]
    
    # Skeleton connections
    skeleton_connections = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('right_shoulder', 'right_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_elbow', 'right_wrist'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'),
        ('right_knee', 'right_ankle'),
    ]
    
    # Draw connections
    for start_point, end_point in skeleton_connections:
        if start_point in landmarks and end_point in landmarks:
            start = landmarks[start_point]
            end = landmarks[end_point]
            
            start_pos = (int(start['x'] * width), int(start['y'] * height))
            end_pos = (int(end['x'] * width), int(end['y'] * height))
            
            cv2.line(image, start_pos, end_pos, (0, 255, 0), 2)
    
    # Draw landmarks
    for landmark in landmarks.values():
        pos = (int(landmark['x'] * width), int(landmark['y'] * height))
        cv2.circle(image, pos, 3, (0, 0, 255), -1)
    
    return image


# ---------------------------
# ====== Prediction Functions
# ---------------------------
def make_undirected_edge_index(edges):
    """Convert list of (u,v) to undirected edge_index tensor [2, num_edges*2]"""
    all_edges = []
    for u, v in edges:
        all_edges.append((u, v))
        all_edges.append((v, u))
    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    return edge_index


def landmarks_to_graph(landmarks: Dict, edge_index: torch.LongTensor) -> Data:
    """Convert landmarks dictionary to PyTorch Geometric Data object"""
    node_features = []
    for name in LANDMARK_NAMES:
        landmark = landmarks.get(name, {"x": 0.0, "y": 0.0, "z": 0.0})
        node_features.append([landmark.get('x', 0.0), landmark.get('y', 0.0), landmark.get('z', 0.0)])
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data


def load_model(model_path: str, device='cpu') -> nn.Module:
    """
    Load the pre-trained GCN model.
    
    Args:
        model_path (str): Path to the model file (.pth)
        device (str): Device to load model on ('cpu' or 'cuda')
        
    Returns:
        nn.Module: Loaded model
    """
    # Instantiate model architecture
    model = SkeletonGCN(
        num_node_features=3,
        hidden_channels=[64, 32, 32, 32, 32],
        num_classes=1,
        dropout_rate=0.3,
        residual=True,
        seed=42
    ).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load state dict
    state = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(state, dict):
        candidate_keys = ['state_dict', 'model_state', 'model_state_dict', 'model']
        model_state = None
        for k in candidate_keys:
            if k in state:
                model_state = state[k]
                break
        
        if model_state is None:
            # Check if it's a raw state_dict
            some_vals = list(state.values())[:5]
            if len(some_vals) > 0 and all(hasattr(v, 'shape') for v in some_vals):
                model_state = state
            else:
                raise RuntimeError(f"Cannot find model weights in checkpoint. Keys: {list(state.keys())}")
    else:
        model_state = state
    
    # Strip 'module.' prefix if present (from DataParallel)
    if isinstance(model_state, dict):
        if any(k.startswith('module.') for k in model_state.keys()):
            new_state = {}
            for k, v in model_state.items():
                new_state[k.replace('module.', '')] = v
            model_state = new_state
    
    # Load state dict
    model.load_state_dict(model_state, strict=False)
    model.eval()
    
    return model


def predict_fall(image_path: str, model: nn.Module, device='cpu', 
                 return_visualization=False) -> Tuple[str, float, Optional[np.ndarray]]:
    """
    Predict whether a person in the image is falling or not.
    
    Args:
        image_path (str): Path to the image file or numpy array
        model (nn.Module): Pre-trained GCN model
        device (str): Device to run inference on
        return_visualization (bool): Whether to return skeleton visualization
        
    Returns:
        Tuple[str, float, Optional[np.ndarray]]: 
            - Prediction label ('fall' or 'not_fall')
            - Confidence probability (0-1)
            - Visualization image (if return_visualization=True)
    """
    # Extract skeleton
    landmarks = extract_skeleton_from_image(image_path)
    
    if landmarks is None:
        return "no_person_detected", 0.0, None
    
    # Create visualization if requested
    visualization = None
    if return_visualization:
        visualization = draw_skeleton_on_image(image_path, landmarks)
    
    # Convert to graph
    edge_index = make_undirected_edge_index(skeleton_edges)
    graph = landmarks_to_graph(landmarks, edge_index)
    
    # Run prediction
    model.eval()
    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, batch)
        out = out.cpu()
        
        if out.size(-1) == 1:
            prob = float(out.squeeze().item())
            # INVERTED: Model output was reversed - high prob means NOT falling
            pred_label = 'not_fall' if prob >= 0.5 else 'fall'
            confidence = prob if prob >= 0.5 else (1 - prob)
        else:
            probs = torch.exp(out)
            prob_val, pred_idx = torch.max(probs, dim=-1)
            pred_label = 'fall' if pred_idx.item() == 1 else 'not_fall'
            confidence = float(prob_val.item())
    
    return pred_label, confidence, visualization


# ---------------------------
# ====== Main Testing Class
# ---------------------------
class FallDetector:
    """
    Fall Detection System
    
    Usage:
        detector = FallDetector(model_path='best.pth')
        label, confidence, viz = detector.predict('image.jpg', visualize=True)
        print(f"Prediction: {label} (confidence: {confidence:.2%})")
    """
    
    def __init__(self, model_path: str = 'best.pth', device: str = None):
        """
        Initialize the fall detector.
        
        Args:
            model_path (str): Path to the trained model file
            device (str): Device to use ('cpu' or 'cuda'). Auto-detect if None.
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading model on {self.device}...")
        self.model = load_model(model_path, self.device)
        print("Model loaded successfully!")
    
    def predict(self, image_path: str, visualize: bool = False) -> Tuple[str, float, Optional[np.ndarray]]:
        """
        Predict fall detection for an image.
        
        Args:
            image_path (str): Path to the image file
            visualize (bool): Whether to return skeleton visualization
            
        Returns:
            Tuple[str, float, Optional[np.ndarray]]:
                - Prediction label ('fall', 'not_fall', or 'no_person_detected')
                - Confidence score (0-1)
                - Visualization image (if visualize=True)
        """
        return predict_fall(image_path, self.model, self.device, visualize)
    
    def predict_batch(self, image_paths: list, visualize: bool = False) -> list:
        """
        Predict fall detection for multiple images.
        
        Args:
            image_paths (list): List of image file paths
            visualize (bool): Whether to return skeleton visualizations
            
        Returns:
            list: List of tuples (label, confidence, visualization)
        """
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, visualize)
            results.append(result)
        return results


# ---------------------------
# ====== CLI Interface ======
# ---------------------------
if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Fall Detection Testing Program')
    parser.add_argument('image', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default='best.pth', help='Path to model file')
    parser.add_argument('--visualize', action='store_true', help='Save skeleton visualization')
    parser.add_argument('--output', type=str, default=None, help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FallDetector(model_path=args.model)
    
    # Run prediction
    print(f"\nAnalyzing image: {args.image}")
    label, confidence, viz = detector.predict(args.image, visualize=args.visualize)
    
    # Print results
    print("\n" + "="*50)
    print(f"PREDICTION: {label.upper()}")
    print(f"CONFIDENCE: {confidence:.2%}")
    print("="*50 + "\n")
    
    # Save visualization if requested
    if args.visualize and viz is not None:
        output_path = args.output or args.image.replace('.', '_skeleton.')
        cv2.imwrite(output_path, viz)
        print(f"Visualization saved to: {output_path}")
    elif args.visualize and viz is None:
        print("Warning: No person detected in the image. Cannot create visualization.")
