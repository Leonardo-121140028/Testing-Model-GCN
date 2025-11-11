# Fall Detection Testing Guide

This guide explains how to use the fall detection testing system.

## Overview

The system uses a Graph Convolutional Network (GCN) trained on skeleton data to detect if a person in an image is falling or not. It consists of:

1. **Skeleton Extraction**: Uses MediaPipe to extract 33 body keypoints
2. **GCN Model**: Processes the skeleton graph to classify fall/not-fall
3. **Web Interface**: Easy-to-use web app for testing

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model File

Make sure `best.pth` is in the project directory.

## Usage Options

### Option 1: Web Interface (Recommended)

The easiest way to test the model with a visual interface.

```bash
python web_app.py
```

Then open your browser to: **http://localhost:5000**

**Features:**
- Drag and drop image upload
- Real-time prediction
- Skeleton visualization
- Confidence scores
- Beautiful UI

### Option 2: Command Line Interface

For quick testing from the terminal.

```bash
python fall_detection_test.py <image_path> [options]
```

**Examples:**

```bash
# Basic prediction
python fall_detection_test.py test_image.jpg

# With skeleton visualization
python fall_detection_test.py test_image.jpg --visualize

# Custom output path
python fall_detection_test.py test_image.jpg --visualize --output result.jpg

# Custom model path
python fall_detection_test.py test_image.jpg --model path/to/model.pth
```

### Option 3: Python API

For integration into your own code.

```python
from fall_detection_test import FallDetector

# Initialize detector
detector = FallDetector(model_path='best.pth')

# Single image prediction
label, confidence, visualization = detector.predict('image.jpg', visualize=True)

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2%}")

# Save visualization
if visualization is not None:
    import cv2
    cv2.imwrite('result.jpg', visualization)

# Batch prediction
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = detector.predict_batch(image_paths, visualize=True)

for img_path, (label, conf, viz) in zip(image_paths, results):
    print(f"{img_path}: {label} ({conf:.2%})")
```

## Output Explanation

### Prediction Labels

- **`fall`**: Person is detected as falling (confidence ≥ 50%)
- **`not_fall`**: Person is standing/normal position (confidence ≥ 50%)
- **`no_person_detected`**: No human pose detected in the image

### Confidence Score

- Range: 0.0 to 1.0 (0% to 100%)
- Higher values indicate more certainty
- Threshold: 0.5 (50%) for binary classification

### Visualization

The skeleton visualization shows:
- **Green lines**: Connections between body joints
- **Red dots**: Detected keypoints (33 landmarks)

## Model Architecture

**SkeletonGCN Details:**
- Input: 33 nodes × 3 features (x, y, z coordinates)
- Hidden layers: [64, 32, 32, 32, 32]
- Output: Binary classification (sigmoid activation)
- Dropout: 0.3
- Pooling: Global mean pooling
- Residual connections: Enabled

**Skeleton Graph:**
- 33 nodes (MediaPipe pose landmarks)
- 30 edges (body connections)
- Undirected graph

## Testing Tips

### Good Test Images

✅ **Recommended:**
- Clear, well-lit images
- Full body visible
- Single person in frame
- High resolution (at least 640×480)
- Person facing camera or side view

❌ **Avoid:**
- Multiple people overlapping
- Heavily occluded poses
- Very low resolution
- Extreme lighting conditions
- Person too far from camera

### Expected Performance

The model was trained on specific fall detection datasets. Performance may vary based on:
- Image quality
- Pose clarity
- Camera angle
- Lighting conditions

## Troubleshooting

### Issue: "No person detected"

**Solutions:**
- Ensure person is clearly visible
- Check if image is too dark/bright
- Verify person's full body is in frame
- Try a different camera angle

### Issue: Model file not found

**Solution:**
```bash
# Verify file exists
ls -la best.pth

# If missing, ensure you have the trained model file
```

### Issue: Import errors

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# For torch-geometric, you may need specific versions
pip install torch-geometric torch-scatter torch-sparse
```

### Issue: Web app not loading

**Solution:**
```bash
# Check if port 5000 is available
lsof -i :5000

# Use different port
# Edit web_app.py, change: app.run(port=5001)
```

## API Reference

### FallDetector Class

```python
class FallDetector:
    def __init__(self, model_path: str, device: str = None)
    def predict(self, image_path: str, visualize: bool = False) -> Tuple[str, float, np.ndarray]
    def predict_batch(self, image_paths: list, visualize: bool = False) -> list
```

### Standalone Functions

```python
# Extract skeleton from image
landmarks = extract_skeleton_from_image(image_path)

# Draw skeleton on image
viz_image = draw_skeleton_on_image(image, landmarks)

# Load model
model = load_model(model_path, device='cpu')

# Predict
label, confidence, viz = predict_fall(image_path, model, device='cpu', return_visualization=True)
```

## File Structure

```
Testing-Model-GCN/
├── best.pth                    # Trained GCN model
├── fall_detection_test.py      # Main testing module
├── web_app.py                  # Flask web interface
├── templates/
│   └── index.html             # Web UI template
├── requirements.txt            # Python dependencies
├── TEST_GUIDE.md              # This file
├── README.md                   # Project overview
├── extract_skeleton.py         # Skeleton extraction (batch)
├── prediction_terbaru.py       # Batch prediction script
└── video_frame_extract.py      # Video frame extraction
```

## Performance Notes

- **CPU**: ~1-2 seconds per image
- **GPU**: ~0.5-1 second per image
- **Memory**: ~500MB with model loaded

## Example Workflow

### Complete Testing Workflow

```bash
# 1. Start web interface
python web_app.py

# 2. Open browser to http://localhost:5000

# 3. Upload test image

# 4. View results:
#    - Prediction label
#    - Confidence score
#    - Original image
#    - Skeleton visualization

# 5. Test another image or close
```

### Batch Processing Workflow

```python
from fall_detection_test import FallDetector
import os

# Initialize
detector = FallDetector('best.pth')

# Get all images from directory
image_dir = 'test_images/'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if f.endswith(('.jpg', '.png'))]

# Process all images
results = detector.predict_batch(image_files, visualize=True)

# Save results
for img_path, (label, conf, viz) in zip(image_files, results):
    print(f"{os.path.basename(img_path)}: {label} ({conf:.2%})")
    
    # Save visualization
    if viz is not None:
        output_path = img_path.replace('.', '_result.')
        cv2.imwrite(output_path, viz)
```

## Support

For issues or questions:
1. Check this guide thoroughly
2. Verify all dependencies are installed
3. Ensure model file (`best.pth`) is present
4. Test with sample images first

## License

This testing system is provided as-is for fall detection research and development.
