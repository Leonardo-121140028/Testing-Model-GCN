# Testing-Model-GCN

Fall Detection System using Graph Convolutional Networks (GCN) with skeleton-based pose estimation.

## 🚀 Quick Start - Single Image Testing

### Web Interface (Easiest)
```bash
pip install -r requirements.txt
python web_app.py
```
Then open: **http://localhost:5000**

### Command Line
```bash
python fall_detection_test.py your_image.jpg --visualize
```

### Python API
```python
from fall_detection_test import FallDetector

detector = FallDetector('best.pth')
label, confidence, viz = detector.predict('image.jpg', visualize=True)
print(f"{label}: {confidence:.2%}")
```

📖 **See [TEST_GUIDE.md](TEST_GUIDE.md) for complete documentation**

---

## 📁 Project Structure

### New Testing System (Single Image)
- **`fall_detection_test.py`** - Main testing module for single images
- **`web_app.py`** - Flask web interface for easy testing
- **`templates/index.html`** - Beautiful web UI
- **`TEST_GUIDE.md`** - Complete testing documentation

### Original Batch Processing Pipeline
- **`video_frame_extract.py`** - Extract frames from videos using datatest_desc.xlsx
- **`extract_skeleton.py`** - Batch skeleton extraction from frames
- **`prediction_terbaru.py`** - Batch prediction with confusion matrix
- **`datatest_desc.xlsx`** - Video segment descriptions

### Model
- **`best.pth`** - Trained GCN model weights

---

## 🎯 Features

### Single Image Testing (NEW)
✅ Upload any image and get instant fall detection  
✅ Web interface with drag-and-drop  
✅ Skeleton visualization overlay  
✅ Confidence scores  
✅ Command-line interface  
✅ Python API for integration  

### Batch Processing (Original)
✅ Video frame extraction  
✅ Batch skeleton extraction  
✅ Batch prediction with metrics  
✅ Confusion matrix generation  

---

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import mediapipe; print('Ready!')"
```

---

## 📊 Model Details

- **Architecture**: Graph Convolutional Network (GCN)
- **Input**: 33 skeleton keypoints (x, y, z coordinates)
- **Output**: Binary classification (fall / not_fall)
- **Backbone**: MediaPipe Pose for skeleton extraction
- **Hidden Layers**: [64, 32, 32, 32, 32]
- **Features**: Residual connections, batch normalization, dropout

---

## 🎨 Usage Examples

### Example 1: Web Testing
```bash
python web_app.py
# Upload image via browser
# Get instant results with visualization
```

### Example 2: CLI Testing
```bash
# Basic prediction
python fall_detection_test.py test.jpg

# With visualization saved
python fall_detection_test.py test.jpg --visualize --output result.jpg
```

### Example 3: Batch Processing
```python
from fall_detection_test import FallDetector

detector = FallDetector('best.pth')
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = detector.predict_batch(images, visualize=True)

for img, (label, conf, viz) in zip(images, results):
    print(f"{img}: {label} ({conf:.2%})")
```

### Example 4: Original Pipeline (Video Processing)
```bash
# 1. Extract frames from video
python video_frame_extract.py

# 2. Extract skeletons from frames
python extract_skeleton.py

# 3. Run batch predictions
python prediction_terbaru.py
```

---

## 📈 Performance

- **Accuracy**: Depends on training data quality
- **Speed**: ~1-2 seconds per image (CPU), ~0.5s (GPU)
- **Input**: Any image with visible human pose
- **Output**: fall / not_fall / no_person_detected

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| No person detected | Ensure full body is visible in image |
| Model not found | Verify `best.pth` exists in directory |
| Import errors | Run `pip install -r requirements.txt` |
| Web app won't start | Check if port 5000 is available |

See [TEST_GUIDE.md](TEST_GUIDE.md) for detailed troubleshooting.

---

## 📚 Documentation

- **[TEST_GUIDE.md](TEST_GUIDE.md)** - Complete testing guide
- **[requirements.txt](requirements.txt)** - Python dependencies
- **Code comments** - Inline documentation in all scripts

---

## 🔄 Workflow Comparison

### New: Single Image Testing
```
Image → Skeleton Extraction → GCN Model → Prediction
(Instant, web-based, user-friendly)
```

### Original: Batch Video Processing
```
Video → Frame Extraction → Skeleton Extraction → Batch Prediction → Metrics
(Batch processing, evaluation-focused)
```

---

## 📝 License

This project is for fall detection research and development.


