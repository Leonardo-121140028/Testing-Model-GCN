"""
Fall Detection Web Application

A Flask-based web interface for fall detection testing.
Upload an image and get instant fall/not-fall prediction with visualization.
"""

import os
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from fall_detection_test import FallDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize detector
detector = None

def init_detector(model_path='best.pth'):
    """Initialize the fall detector"""
    global detector
    if detector is None:
        print("Initializing Fall Detector...")
        detector = FallDetector(model_path=model_path)
        print("Detector ready!")

def image_to_base64(image_array):
    """Convert numpy array image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image_array)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Save temporarily
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
        cv2.imwrite(temp_path, image)
        
        # Run prediction
        label, confidence, viz = detector.predict(temp_path, visualize=True)
        
        # Convert images to base64
        original_b64 = image_to_base64(image)
        viz_b64 = image_to_base64(viz) if viz is not None else None
        
        # Prepare response
        response = {
            'prediction': label,
            'confidence': float(confidence),
            'original_image': original_b64,
            'visualization': viz_b64
        }
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model_loaded': detector is not None})

if __name__ == '__main__':
    # Initialize detector
    init_detector()
    
    # Run app
    print("\n" + "="*60)
    print("Fall Detection Web App Starting...")
    print("Open your browser and go to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5123)
