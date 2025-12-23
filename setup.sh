#!/bin/bash

# CIFAR-10 Dashboard Quick Start Script

echo "=================================="
echo "CIFAR-10 Dashboard Setup"
echo "=================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
echo "Python 3 found: $(python3 --version)"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi
echo "Node.js found: $(node --version)"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Install frontend dependencies
echo ""
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p results logs

# Copy environment file
echo ""
echo "Setting up environment..."
cd frontend
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file"
else
    echo ".env file already exists"
fi
cd ..

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To start the application:"
echo ""
echo "1. Start the backend API:"
echo "   cd backend/api && python app.py"
echo ""
echo "2. In a new terminal, start the frontend:"
echo "   cd frontend && npm run dev"
echo ""
echo "3. Open your browser to:"
echo "   http://localhost:4321"
echo ""
echo "=================================="
