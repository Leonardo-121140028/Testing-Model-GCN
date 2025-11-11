# Quick Start Guide - Fall Detection System

## 🎯 What This System Does

Upload an image of a person → System detects skeleton → Predicts if person is falling or not

## 📦 Installation (One-Time Setup)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- PyTorch Geometric (graph neural networks)
- OpenCV (image processing)
- MediaPipe (skeleton detection)
- Flask (web interface)
- Other utilities

**Note**: Installation may take 5-10 minutes depending on your internet speed.

### Step 2: Verify Installation

```bash
python test_installation.py
```

You should see all tests pass ✅

## 🚀 Running the System

### Option 1: Web Interface (Recommended for Testing)

```bash
python web_app.py
```

Then open your browser to: **http://localhost:5000**

**What you'll see:**
1. A beautiful upload interface
2. Drag and drop or click to upload an image
3. Click "Analyze Image"
4. Get results:
   - Prediction: FALL or NOT FALL
   - Confidence percentage
   - Original image
   - Skeleton visualization

### Option 2: Command Line (Quick Testing)

```bash
# Test a single image
python fall_detection_test.py path/to/your/image.jpg

# With visualization saved
python fall_detection_test.py path/to/your/image.jpg --visualize
```

**Output:**
```
==================================================
PREDICTION: NOT_FALL
CONFIDENCE: 87.50%
==================================================
```

### Option 3: Python Script (For Integration)

Create a file `my_test.py`:

```python
from fall_detection_test import FallDetector

# Initialize detector (loads model)
detector = FallDetector('best.pth')

# Test an image
label, confidence, visualization = detector.predict('test_image.jpg', visualize=True)

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2%}")

# Save visualization
if visualization is not None:
    import cv2
    cv2.imwrite('result.jpg', visualization)
    print("Visualization saved to result.jpg")
```

Run it:
```bash
python my_test.py
```

## 📸 Test Image Requirements

**Good images:**
- ✅ Clear, well-lit photo
- ✅ Full body visible
- ✅ Single person
- ✅ Resolution at least 640x480

**Avoid:**
- ❌ Multiple overlapping people
- ❌ Very dark or bright images
- ❌ Person too far away
- ❌ Heavily cropped (missing body parts)

## 🎨 Understanding the Results

### Prediction Labels

1. **`fall`** - Person is detected as falling
   - Body angle suggests falling motion
   - Confidence ≥ 50%

2. **`not_fall`** - Person is standing/normal
   - Upright posture detected
   - Confidence ≥ 50%

3. **`no_person_detected`** - No human pose found
   - Image may be unclear
   - No person in frame
   - Person too small/far

### Confidence Score

- **80-100%**: Very confident prediction
- **60-80%**: Confident prediction
- **50-60%**: Uncertain (borderline case)

### Skeleton Visualization

The visualization shows:
- **Green lines**: Body connections (bones)
- **Red dots**: Joint positions (keypoints)
- 33 keypoints total (MediaPipe landmarks)

## 🔧 Troubleshooting

### Problem: "Module not found" errors

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "Model file not found"

**Solution:**
Ensure `best.pth` is in the same directory as the scripts.

```bash
ls -la best.pth
# Should show: best.pth (around 100KB)
```

### Problem: "No person detected" for valid images

**Possible causes:**
1. Image is too dark/bright
2. Person is too small in frame
3. Body is heavily occluded
4. Image quality is poor

**Solutions:**
- Use better lighting
- Get closer to subject
- Ensure full body is visible
- Use higher resolution image

### Problem: Web app won't start

**Check if port 5000 is in use:**
```bash
lsof -i :5000
```

**Use different port:**
Edit `web_app.py`, change the last line:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Changed to 5001
```

### Problem: Slow predictions

**CPU mode is slower (1-2 seconds per image)**

If you have NVIDIA GPU:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

If True, the system will automatically use GPU (much faster).

## 📊 Example Workflow

### Complete Testing Session

```bash
# 1. Install (one time)
pip install -r requirements.txt

# 2. Verify installation
python test_installation.py

# 3. Start web interface
python web_app.py

# 4. Open browser
# Go to: http://localhost:5000

# 5. Test images
# - Upload test image
# - View prediction
# - Check skeleton visualization
# - Try more images

# 6. Done! Press Ctrl+C to stop server
```

## 🎯 Use Cases

### 1. Safety Monitoring
Monitor elderly people or patients for fall detection

### 2. Sports Analysis
Analyze athletic movements and falls

### 3. Security Systems
Detect accidents or incidents in surveillance

### 4. Research
Study human pose and fall patterns

## 📁 File Overview

```
Testing-Model-GCN/
├── best.pth                    ← Trained model (required)
├── fall_detection_test.py      ← Main testing module
├── web_app.py                  ← Web interface
├── templates/index.html        ← Web UI
├── requirements.txt            ← Dependencies
├── QUICK_START.md             ← This file
├── TEST_GUIDE.md              ← Detailed guide
└── README.md                   ← Project overview
```

## 🆘 Getting Help

1. **Read the guides:**
   - QUICK_START.md (this file) - Basic usage
   - TEST_GUIDE.md - Detailed documentation
   - README.md - Project overview

2. **Test installation:**
   ```bash
   python test_installation.py
   ```

3. **Check model file:**
   ```bash
   ls -la best.pth
   ```

4. **Verify Python version:**
   ```bash
   python --version
   # Should be Python 3.8 or higher
   ```

## 🎉 Success Checklist

Before testing, ensure:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Model file `best.pth` present
- [ ] Test installation passed
- [ ] Have test images ready

Then you're ready to test! 🚀

## 💡 Tips

1. **Start with web interface** - Easiest way to test
2. **Use good quality images** - Better results
3. **Test multiple images** - Understand model behavior
4. **Check confidence scores** - Gauge prediction reliability
5. **Save visualizations** - Useful for analysis

## 📝 Next Steps

After testing:
1. Try different images (standing, falling, sitting)
2. Analyze confidence scores
3. Compare skeleton visualizations
4. Integrate into your application (use Python API)
5. Batch process multiple images

---

**Ready to start?**

```bash
python web_app.py
```

Then open: http://localhost:5000

Happy testing! 🎉
