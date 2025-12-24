"""
Flask API for VisionAI Power Training and Inference
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
import tensorflow as tf

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import CIFAR10DataLoader
from model import CIFAR10Model, VisionAIModel
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
        data_loader = CIFAR10DataLoader(batch_size=32)
        data_loader.load_data(use_augmentation=True)
    return data_loader


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "VisionAI Power Engine API"})


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


@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start powerful AI model training"""
    global training_thread, training_status
    
    if training_status["is_training"]:
        return jsonify({"error": "Training already in progress"}), 400
    
    # Get configuration from request
    config = request.json or {}
    epochs = config.get('epochs', 20)
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.0001)
    
    def train_model():
        global training_status
        try:
            training_status = {"is_training": True, "progress": 0, "message": "Initializing Power AI Base..."}
            
            loader = CIFAR10DataLoader(batch_size=batch_size)
            loader.load_data(use_augmentation=True)
            
            # Using the new Power Model
            model = CIFAR10Model(num_classes=10)
            model.build(input_shape=loader.input_shape)
            model.compile(optimizer="adam", learning_rate=learning_rate)
            
            trainer_config = {
                "epochs": epochs,
                "batch_size": batch_size,
                "model_name": "visionai-power-model",
                "log_dir": LOGS_DIR,
                "model_dir": RESULTS_DIR
            }
            
            trainer = ModelTrainer(model, loader, trainer_config)
            training_status["message"] = "VisionAI is learning..."
            
            trainer.train()
            
            training_status = {
                "is_training": False,
                "progress": 100,
                "message": "AI Training completed successfully!"
            }
        except Exception as e:
            training_status = {
                "is_training": False,
                "progress": 0,
                "message": f"AI Training failed: {str(e)}"
            }
    
    training_thread = threading.Thread(target=train_model)
    training_thread.start()
    return jsonify({"message": "Power training started", "status": training_status})


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make high-accuracy prediction on uploaded image"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        from tensorflow.keras.models import load_model
        
        # Load the latest best model
        model_name = request.form.get('model', 'visionai-power-model-best.h5')
        model_path = os.path.join(RESULTS_DIR, model_name)
        
        if not os.path.exists(model_path):
            return jsonify({"error": "Power Model not found. Did you train it?"}), 404
        
        model = load_model(model_path)
        
        # Process image with AI standard (224x224 + MobileNet preprocessing)
        image_file = request.files['image']
        img = Image.open(image_file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        
        # AI Normalization (MobileNet style: -1 to 1)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class])
        
        loader = initialize_data_loader()
        probabilities = {
            loader.get_class_name(i): float(prediction[0][i])
            for i in range(len(prediction[0]))
        }
        
        return jsonify({
            "predicted_class": predicted_class,
            "predicted_label": loader.get_class_name(predicted_class),
            "confidence": confidence,
            "probabilities": probabilities,
            "engine": "VisionAI Power Hub"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/training/status')
def training_status_endpoint():
    return jsonify(training_status)


@app.route('/api/training/progress')
def training_progress():
    progress_file = os.path.join(RESULTS_DIR, "training_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"epochs": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []})


if __name__ == '__main__':
    print("="*60)
    print("VisionAI Power Engine API - Python Backend")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
