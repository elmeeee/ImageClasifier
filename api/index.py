import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from backend.api.app import app
except ImportError as e:
    # Fallback: create a simple error response app
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/api/<path:path>')
    @app.route('/api/')
    def error_handler(path=''):
        return jsonify({
            "error": "Import failed",
            "message": str(e),
            "path": sys.path,
            "cwd": os.getcwd(),
            "files": os.listdir(os.getcwd())
        }), 500
