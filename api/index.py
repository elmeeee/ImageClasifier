"""
Vercel serverless function entry point
"""
from api.app import app

# Vercel will automatically handle WSGI for Flask apps
