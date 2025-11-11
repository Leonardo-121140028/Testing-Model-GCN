# Improving Fall Detection Accuracy

## 🔍 Problem Identified

The model is predicting "fall" for images showing people standing (not falling). This suggests one of several issues:

1. **Label mapping is inverted** (most likely)
2. **Model was trained with incorrect labels**
3. **Threshold needs adjustment**
4. **Model needs retraining with better data**

---

## 📊 Step 1: Diagnose the Problem

First, run the diagnosis script to understand the model's behavior:

```bash
python diagnose_model.py
```

This will show you:
- Raw model outputs
- Current predictions
- Inverted predictions
- Which logic makes more sense

---

## 🚀 Quick Fixes (Try These First)

### Solution 1: Invert the Prediction Logic (RECOMMENDED)

The model output might be inverted. Let's fix the prediction logic:

**Current logic (line 365 in fall_detection_test.py):**
```python
pred_label = 'fall' if prob >= 0.5 else 'not_fall'
```

**Change to:**
```python
pred_label = 'not_fall' if prob >= 0.5 else 'fall'  # INVERTED
```

**How to apply:**

1. Open `fall_detection_test.py`
2. Go to line 365
3. Change the prediction logic
4. Save and restart the web app

---

### Solution 2: Adjust the Threshold

If inverting doesn't work, try adjusting the threshold:

**Current threshold: 0.5**

Try different values:
```python
# More conservative (fewer fall predictions)
pred_label = 'fall' if prob >= 0.7 else 'not_fall'

# More sensitive (more fall predictions)
pred_label = 'fall' if prob >= 0.3 else 'not_fall'
```

---

### Solution 3: Add Confidence Filtering

Only trust predictions with high confidence:

```python
if out.size(-1) == 1:
    prob = float(out.squeeze().item())
    
    # Add confidence threshold
    if prob > 0.8:  # Very confident it's fall
        pred_label = 'fall'
        confidence = prob
    elif prob < 0.2:  # Very confident it's not fall
        pred_label = 'not_fall'
        confidence = 1 - prob
    else:  # Uncertain
        pred_label = 'uncertain'
        confidence = 0.5
```

---

## 🔧 Advanced Solutions

### Solution 4: Add Feature Normalization

Normalize skeleton coordinates before feeding to model:

```python
def normalize_skeleton(landmarks):
    """
    Normalize skeleton to be scale and position invariant
    """
    # Extract all coordinates
    coords = []
    for name in LANDMARK_NAMES:
        lm = landmarks.get(name, {'x': 0, 'y': 0, 'z': 0})
        coords.append([lm['x'], lm['y'], lm['z']])
    
    coords = np.array(coords)
    
    # Center at origin (subtract mean)
    coords = coords - coords.mean(axis=0)
    
    # Scale to unit variance
    std = coords.std()
    if std > 0:
        coords = coords / std
    
    # Update landmarks
    normalized = {}
    for i, name in enumerate(LANDMARK_NAMES):
        normalized[name] = {
            'x': float(coords[i, 0]),
            'y': float(coords[i, 1]),
            'z': float(coords[i, 2])
        }
    
    return normalized
```

Add this before creating the graph:
```python
landmarks = normalize_skeleton(landmarks)
graph = landmarks_to_graph(landmarks, edge_index)
```

---

### Solution 5: Add Angle-Based Features

Add body angle as an additional feature:

```python
def calculate_body_angle(landmarks):
    """
    Calculate the angle of the body from vertical
    Returns angle in degrees (0 = upright, 90 = horizontal)
    """
    # Get key points
    if 'left_shoulder' not in landmarks or 'left_hip' not in landmarks:
        return 0
    
    shoulder = landmarks['left_shoulder']
    hip = landmarks['left_hip']
    
    # Calculate angle
    dy = hip['y'] - shoulder['y']
    dx = hip['x'] - shoulder['x']
    
    angle = np.abs(np.arctan2(dx, dy) * 180 / np.pi)
    return angle

def is_falling_by_angle(landmarks, threshold=45):
    """
    Simple rule-based check using body angle
    """
    angle = calculate_body_angle(landmarks)
    return angle > threshold  # If body is tilted > 45°, likely falling
```

Combine with model prediction:
```python
# Get model prediction
model_pred = 'fall' if prob >= 0.5 else 'not_fall'

# Get angle-based prediction
angle_pred = 'fall' if is_falling_by_angle(landmarks) else 'not_fall'

# Combine (both must agree)
if model_pred == angle_pred:
    pred_label = model_pred
    confidence = prob if prob >= 0.5 else (1 - prob)
else:
    pred_label = 'uncertain'
    confidence = 0.5
```

---

### Solution 6: Ensemble with Multiple Checks

Use multiple heuristics:

```python
def advanced_fall_detection(landmarks, model_prob):
    """
    Combine model prediction with rule-based checks
    """
    checks = []
    
    # Check 1: Model prediction
    model_says_fall = model_prob >= 0.5
    checks.append(model_says_fall)
    
    # Check 2: Body angle
    angle = calculate_body_angle(landmarks)
    angle_says_fall = angle > 45
    checks.append(angle_says_fall)
    
    # Check 3: Head position (if head is low, might be falling)
    if 'nose' in landmarks and 'left_ankle' in landmarks:
        head_y = landmarks['nose']['y']
        ankle_y = landmarks['left_ankle']['y']
        head_low = (ankle_y - head_y) < 0.3  # Head close to feet level
        checks.append(head_low)
    
    # Check 4: Hip height (if hips are low, might be on ground)
    if 'left_hip' in landmarks:
        hip_y = landmarks['left_hip']['y']
        hip_low = hip_y > 0.7  # Hip in lower part of image
        checks.append(hip_low)
    
    # Majority vote
    fall_votes = sum(checks)
    total_votes = len(checks)
    
    if fall_votes >= total_votes * 0.6:  # 60% threshold
        return 'fall', fall_votes / total_votes
    else:
        return 'not_fall', 1 - (fall_votes / total_votes)
```

---

## 🎯 Recommended Approach

### Step-by-Step Fix

1. **Run diagnosis:**
   ```bash
   python diagnose_model.py
   ```

2. **Try Solution 1 (Invert logic):**
   - Edit `fall_detection_test.py` line 365
   - Change: `pred_label = 'not_fall' if prob >= 0.5 else 'fall'`
   - Test with web app

3. **If still wrong, try Solution 2 (Adjust threshold):**
   - Try threshold = 0.3, 0.4, 0.6, 0.7
   - Find the best value

4. **If still issues, implement Solution 5 (Angle-based):**
   - Add angle calculation
   - Combine with model prediction

5. **Last resort: Retrain the model** (see below)

---

## 🔄 Long-Term Solution: Retrain the Model

If quick fixes don't work, you need to retrain with correct labels.

### Check Training Data Labels

1. **Verify your training data:**
   ```bash
   # Check the skeleton data directories
   ls test/data-skeleton/fall/
   ls test/data-skeleton/not_fall/
   ```

2. **Ensure labels are correct:**
   - `fall/` directory should contain falling poses
   - `not_fall/` directory should contain standing/normal poses

3. **If labels are swapped, fix them:**
   ```bash
   # Backup first
   cp -r test/data-skeleton test/data-skeleton_backup
   
   # Swap directories
   mv test/data-skeleton/fall test/data-skeleton/temp
   mv test/data-skeleton/not_fall test/data-skeleton/fall
   mv test/data-skeleton/temp test/data-skeleton/not_fall
   ```

4. **Retrain the model** with correct labels

---

## 📝 Implementation Example

Here's a complete fixed version of the prediction function:

```python
def predict_fall_improved(image_path: str, model: nn.Module, device='cpu', 
                          return_visualization=False):
    """
    Improved fall prediction with multiple checks
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
    
    # Run model prediction
    model.eval()
    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
    
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, batch)
        out = out.cpu()
        prob = float(out.squeeze().item())
    
    # SOLUTION 1: Invert if needed (test this first)
    # pred_label = 'not_fall' if prob >= 0.5 else 'fall'  # INVERTED
    
    # SOLUTION 5: Add angle-based check
    angle = calculate_body_angle(landmarks)
    
    # Combine model and angle
    model_says_fall = prob >= 0.5
    angle_says_fall = angle > 45
    
    if model_says_fall and angle_says_fall:
        pred_label = 'fall'
        confidence = (prob + (angle / 90)) / 2
    elif not model_says_fall and not angle_says_fall:
        pred_label = 'not_fall'
        confidence = ((1 - prob) + (1 - angle / 90)) / 2
    else:
        # Disagreement - use angle as tiebreaker
        pred_label = 'fall' if angle_says_fall else 'not_fall'
        confidence = 0.6  # Lower confidence for disagreement
    
    return pred_label, confidence, visualization

def calculate_body_angle(landmarks):
    """Calculate body tilt angle"""
    if 'left_shoulder' not in landmarks or 'left_hip' not in landmarks:
        return 0
    
    shoulder = landmarks['left_shoulder']
    hip = landmarks['left_hip']
    
    dy = hip['y'] - shoulder['y']
    dx = hip['x'] - shoulder['x']
    
    angle = np.abs(np.arctan2(dx, dy) * 180 / np.pi)
    return angle
```

---

## 🧪 Testing Your Fix

### Test Script

Create `test_fix.py`:

```python
from fall_detection_test import FallDetector
import os

detector = FallDetector('best.pth')

# Test with your images
test_images = {
    'standing_person.jpg': 'not_fall',  # Expected
    'falling_person.jpg': 'fall',       # Expected
}

print("Testing predictions:")
print("="*60)

for img_path, expected in test_images.items():
    if not os.path.exists(img_path):
        print(f"Skipping {img_path} (not found)")
        continue
    
    label, conf, _ = detector.predict(img_path)
    
    status = "✅ CORRECT" if label == expected else "❌ WRONG"
    print(f"\n{img_path}:")
    print(f"  Expected: {expected}")
    print(f"  Predicted: {label} ({conf:.2%})")
    print(f"  Status: {status}")
```

Run:
```bash
python test_fix.py
```

---

## 📊 Evaluation Metrics

After applying fixes, evaluate performance:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Collect predictions
y_true = []  # True labels
y_pred = []  # Predicted labels

for img, true_label in test_set:
    pred_label, _, _ = detector.predict(img)
    y_true.append(1 if true_label == 'fall' else 0)
    y_pred.append(1 if pred_label == 'fall' else 0)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")
```

---

## ✅ Summary

### Quick Fixes (Try in order):
1. ✅ Run `diagnose_model.py`
2. ✅ Invert prediction logic (line 365)
3. ✅ Adjust threshold (0.3 to 0.7)
4. ✅ Add angle-based check
5. ✅ Implement ensemble method

### Long-term:
6. ✅ Verify training data labels
7. ✅ Retrain model if needed
8. ✅ Collect more diverse training data
9. ✅ Use data augmentation
10. ✅ Try different model architectures

---

## 🆘 Need Help?

1. Run diagnosis: `python diagnose_model.py`
2. Check the output and see which prediction makes sense
3. Apply the appropriate solution
4. Test with multiple images
5. Evaluate performance

Good luck! 🚀

<function_calls>
<invoke name="read_file">
<parameter name="file_path">/Users/feedloop/Documents/Testing-Model-GCN/fall_detection_test.py
