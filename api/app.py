"""
Simplified Flask API for Vercel deployment
Only includes lightweight endpoints that work in serverless environment
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# CIFAR-10 class names
CATEGORIES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "CIFAR-10 Training API",
        "environment": "vercel-serverless"
    })

@app.route('/api/dataset/info')
def dataset_info():
    """Get dataset information"""
    return jsonify({
        "num_train_samples": 50000,
        "num_test_samples": 10000,
        "input_shape": [32, 32, 3],
        "num_classes": 10,
        "categories": CATEGORIES,
        "note": "Full dataset operations require local environment"
    })

@app.route('/api/training/status')
def training_status_endpoint():
    """Get current training status"""
    return jsonify({
        "is_training": False,
        "progress": 0,
        "message": "Training not available in serverless environment. Please run locally.",
        "note": "Use 'python backend/train.py' for local training"
    })

@app.route('/api/models/list')
def list_models():
    """List available trained models"""
    return jsonify({
        "models": [],
        "note": "Model storage not available in serverless environment"
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint - placeholder for now"""
    return jsonify({
        "error": "Prediction not available in serverless environment",
        "note": "TensorFlow models are too large for Vercel serverless functions",
        "suggestion": "Deploy the full backend on a platform like Railway, Render, or Fly.io"
    }), 501

@app.route('/api/info')
def api_info():
    """API information and available endpoints"""
    return jsonify({
        "name": "CIFAR-10 API (Serverless)",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "dataset_info": "/api/dataset/info",
            "training_status": "/api/training/status",
            "models_list": "/api/models/list",
            "predict": "/api/predict (not implemented)",
            "info": "/api/info"
        },
        "limitations": [
            "No model training (use local environment)",
            "No dataset downloads (too large for serverless)",
            "No model inference (TensorFlow too heavy for serverless)",
            "10 second timeout limit",
            "2048 MB memory limit"
        ],
        "recommendations": [
            "Use this API for frontend development and testing",
            "Deploy full backend on Railway/Render/Fly.io for production",
            "Run 'python backend/train.py' locally for model training"
        ]
    })

# Catch-all route for undefined API endpoints
@app.route('/api/<path:path>')
def catch_all(path):
    """Handle undefined routes"""
    return jsonify({
        "error": "Endpoint not found",
        "path": f"/api/{path}",
        "available_endpoints": [
            "/api/health",
            "/api/info",
            "/api/dataset/info",
            "/api/training/status",
            "/api/models/list"
        ]
    }), 404

# For local development
if __name__ == '__main__':
    print("="*60)
    print("CIFAR-10 API (Serverless Version)")
    print("="*60)
    print("\nAPI running at: http://localhost:5000")
    print("Health check: http://localhost:5000/api/health")
    print("API info: http://localhost:5000/api/info")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
