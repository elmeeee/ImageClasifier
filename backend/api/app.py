"""
Flask API for CIFAR-10 model training and inference
Designed to work with Astro frontend and deploy on Vercel
"""
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import sys
import json
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import threading

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import CIFAR10DataLoader
from model import CIFAR10Model
from trainer import ModelTrainer

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Global variables
data_loader = None
training_thread = None
training_status = {"is_training": False, "progress": 0, "message": ""}


def initialize_data_loader():
    """Initialize data loader on first request"""
    global data_loader
    if data_loader is None:
        data_loader = CIFAR10DataLoader(batch_size=64)
        data_loader.load_data(use_augmentation=True)
    return data_loader


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "CIFAR-10 Training API"})


@app.route('/api/dataset/info')
def dataset_info():
    """Get dataset information"""
    loader = initialize_data_loader()
    return jsonify({
        "num_train_samples": loader.num_train_examples,
        "num_test_samples": loader.num_test_examples,
        "input_shape": loader.input_shape,
        "num_classes": len(loader.CATEGORIES),
        "categories": loader.CATEGORIES
    })


@app.route('/api/dataset/samples')
def dataset_samples():
    """Get sample images from dataset"""
    loader = initialize_data_loader()
    num_samples = int(request.args.get('num_samples', 25))
    from_test = request.args.get('from_test', 'true').lower() == 'true'
    
    images, labels = loader.get_sample_images(num_samples, from_test)
    
    # Convert images to base64
    samples = []
    for img, label in zip(images, labels):
        # Convert to PIL Image
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        
        # Convert to base64
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        samples.append({
            "image": f"data:image/png;base64,{img_str}",
            "label": int(label),
            "label_name": loader.get_class_name(label)
        })
    
    return jsonify({"samples": samples})


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start model training"""
    global training_thread, training_status
    
    if training_status["is_training"]:
        return jsonify({"error": "Training already in progress"}), 400
    
    # Get configuration from request
    config = request.json or {}
    epochs = config.get('epochs', 30)
    batch_size = config.get('batch_size', 64)
    learning_rate = config.get('learning_rate', 0.001)
    
    # Start training in background thread
    def train_model():
        global training_status
        try:
            training_status = {"is_training": True, "progress": 0, "message": "Initializing..."}
            
            # Setup
            loader = CIFAR10DataLoader(batch_size=batch_size)
            loader.load_data(use_augmentation=True)
            
            model = CIFAR10Model(num_classes=10)
            model.build(input_shape=loader.input_shape)
            model.compile(optimizer="adam", learning_rate=learning_rate)
            
            trainer_config = {
                "epochs": epochs,
                "batch_size": batch_size,
                "model_name": "cifar10-cnn-v2",
                "log_dir": LOGS_DIR,
                "model_dir": RESULTS_DIR
            }
            
            trainer = ModelTrainer(model, loader, trainer_config)
            training_status["message"] = "Training started..."
            
            # Train
            trainer.train()
            
            training_status = {
                "is_training": False,
                "progress": 100,
                "message": "Training completed successfully!"
            }
        except Exception as e:
            training_status = {
                "is_training": False,
                "progress": 0,
                "message": f"Training failed: {str(e)}"
            }
    
    training_thread = threading.Thread(target=train_model)
    training_thread.start()
    
    return jsonify({"message": "Training started", "status": training_status})


@app.route('/api/training/status')
def training_status_endpoint():
    """Get current training status"""
    return jsonify(training_status)


@app.route('/api/training/progress')
def training_progress():
    """Get training progress data"""
    progress_file = os.path.join(RESULTS_DIR, "training_progress.json")
    
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"epochs": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []})


@app.route('/api/training/summary')
def training_summary():
    """Get training summary"""
    summary_file = os.path.join(RESULTS_DIR, "training_summary.json")
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "No training summary available"}), 404


@app.route('/api/models/list')
def list_models():
    """List available trained models"""
    if not os.path.exists(RESULTS_DIR):
        return jsonify({"models": []})
    
    models = []
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith('.h5'):
            filepath = os.path.join(RESULTS_DIR, filename)
            models.append({
                "name": filename,
                "size": os.path.getsize(filepath),
                "modified": os.path.getmtime(filepath)
            })
    
    # Sort by modification time (newest first)
    models.sort(key=lambda x: x['modified'], reverse=True)
    
    return jsonify({"models": models})


@app.route('/api/models/download/<filename>')
def download_model(filename):
    """Download a trained model"""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    if not os.path.exists(filepath) or not filename.endswith('.h5'):
        return jsonify({"error": "Model not found"}), 404
    
    return send_file(filepath, as_attachment=True, download_name=filename)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        from tensorflow.keras.models import load_model
        
        # Load model
        model_name = request.form.get('model', 'cifar10-cnn-v2-best.h5')
        model_path = os.path.join(RESULTS_DIR, model_name)
        
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found"}), 404
        
        model = load_model(model_path)
        
        # Process image
        image_file = request.files['image']
        img = Image.open(image_file.stream)
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0
        
        # Make prediction
        prediction = model.predict(img_array.reshape(1, 32, 32, 3), verbose=0)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class])
        
        loader = initialize_data_loader()
        
        # Get all probabilities
        probabilities = {
            loader.get_class_name(i): float(prediction[0][i])
            for i in range(len(prediction[0]))
        }
        
        return jsonify({
            "predicted_class": predicted_class,
            "predicted_label": loader.get_class_name(predicted_class),
            "confidence": confidence,
            "probabilities": probabilities
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# For local development
if __name__ == '__main__':
    print("="*60)
    print("CIFAR-10 Training API")
    print("="*60)
    print("\nAPI running at: http://localhost:5000")
    print("Health check: http://localhost:5000/api/health")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
