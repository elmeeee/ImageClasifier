# CIFAR-10 Image Classification Dashboard

A modern, full-stack image classification system with a beautiful web dashboard for training, monitoring, and deploying CIFAR-10 models. Built with **Astro**, **TensorFlow**, and **Flask**, optimized for deployment on **Vercel**.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Astro](https://img.shields.io/badge/Astro-4.0+-purple)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)

## Features

### Core Functionality
- **Dataset Explorer** - Visualize CIFAR-10 training and test samples
- **Model Training** - Train CNN models with customizable hyperparameters
- **Real-time Monitoring** - Track training progress with interactive charts
- **Model Management** - Download and manage trained models
- **Predictions** - Test models with custom image uploads

### Modern UI/UX
- **Premium Dark Theme** with glassmorphism effects
- **Smooth Animations** and micro-interactions
- **Responsive Design** for all devices
- **Real-time Updates** with WebSocket-like polling
- **Interactive Charts** using Chart.js

### Technical Highlights
- **Modular Architecture** with clean separation of concerns
- **Improved CNN** with BatchNormalization layers
- **Data Augmentation** for better model generalization
- **Automatic Callbacks** (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- **Progress Logging** for web dashboard integration
- **Vercel Deployment** ready with serverless functions

## Project Structure

```
ImageClasifier/
├── frontend/                 # Astro frontend application
│   ├── src/
│   │   ├── components/      # Astro components
│   │   │   ├── DatasetExplorer.astro
│   │   │   ├── TrainingPanel.astro
│   │   │   ├── ProgressMonitor.astro
│   │   │   ├── ModelsManager.astro
│   │   │   └── PredictionPanel.astro
│   │   ├── layouts/         # Page layouts
│   │   │   └── Layout.astro
│   │   ├── pages/           # Routes
│   │   │   └── index.astro
│   │   └── scripts/         # Client-side JavaScript
│   │       └── main.js
│   ├── astro.config.mjs     # Astro configuration
│   └── package.json
│
├── backend/                  # Python backend
│   ├── api/                 # Flask API
│   │   └── app.py          # Main API server
│   ├── model.py            # CNN model architecture
│   ├── data_loader.py      # Dataset loading & preprocessing
│   ├── trainer.py          # Training utilities
│   ├── train.py            # Training script
│   └── test.py             # Testing & evaluation
│
├── api/                     # Vercel serverless functions
│   └── index.py            # API entry point
│
├── results/                 # Trained models & logs
├── logs/                    # TensorBoard logs
├── vercel.json             # Vercel deployment config
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

### Prerequisites
- **Python 3.10-3.12** (3.10 recommended for Vercel)
- **Node.js 18+**
- **npm or yarn**

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ImageClasifier
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### 4. Setup Environment Variables
```bash
cd frontend
cp .env.example .env
# Edit .env if needed for custom API URL
cd ..
```

## Usage

### Local Development

#### Option 1: Run Backend and Frontend Separately

**Terminal 1 - Start Backend API:**
```bash
cd backend/api
python app.py
```
The API will run on `http://localhost:5000`

**Terminal 2 - Start Frontend Dev Server:**
```bash
cd frontend
npm run dev
```
The frontend will run on `http://localhost:4321`

#### Option 2: Train Model Directly (No UI)
```bash
cd backend
python train.py
```

#### Option 3: Test Trained Model
```bash
cd backend
python test.py
```

### Production Deployment on Vercel

#### 1. Install Vercel CLI
```bash
npm install -g vercel
```

#### 2. Deploy to Vercel
```bash
vercel
```

Follow the prompts to deploy. Vercel will automatically:
- Build the Astro frontend
- Deploy Python API as serverless functions
- Configure routing between frontend and API

#### 3. Set Environment Variables (Optional)
In Vercel dashboard, add:
- `PUBLIC_API_URL` - Your API endpoint (auto-configured)

## Model Architecture

### Enhanced CNN with BatchNormalization

```
Input (32x32x3)
    ↓
[Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)]
    ↓
[Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)]
    ↓
[Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)]
    ↓
[Flatten → Dense(1024) → BatchNorm → ReLU → Dropout(0.5)]
    ↓
Dense(10, softmax)
```

### Training Features
- **Optimizer**: Adam with configurable learning rate
- **Data Augmentation**: Random flips, brightness, contrast
- **Callbacks**:
  - ModelCheckpoint (save best model)
  - EarlyStopping (prevent overfitting)
  - ReduceLROnPlateau (adaptive learning rate)
  - TensorBoard logging
  - Custom progress tracking for web UI

## UI Components

### 1. Dataset Explorer
- View dataset statistics
- Browse 10 CIFAR-10 categories
- Visualize random samples from train/test sets
- Interactive sample grid with hover effects

### 2. Training Panel
- Configure epochs, batch size, learning rate
- Start training with one click
- Real-time status updates
- Auto-redirect to progress monitor

### 3. Progress Monitor
- Live training/validation accuracy charts
- Live training/validation loss charts
- Learning rate schedule visualization
- Training summary with key metrics

### 4. Models Manager
- List all trained models
- View model size and modification date
- One-click model download
- Automatic model selection for predictions

### 5. Prediction Panel
- Drag & drop image upload
- Select model for inference
- View prediction with confidence score
- See probability distribution for all classes

## 🔧 Configuration

### Training Configuration
Edit in `backend/train.py`:
```python
config = {
    "batch_size": 64,        # Batch size for training
    "epochs": 30,            # Number of training epochs
    "learning_rate": 0.001,  # Initial learning rate
    "use_augmentation": True, # Enable data augmentation
    "model_name": "cifar10-cnn-v2",
    "log_dir": "logs",
    "model_dir": "results"
}
```

### API Configuration
Edit in `backend/api/app.py`:
```python
RESULTS_DIR = "results"  # Model storage directory
LOGS_DIR = "logs"        # TensorBoard logs directory
```

### Frontend Configuration
Edit in `frontend/.env`:
```env
PUBLIC_API_URL=http://localhost:5000/api  # API endpoint
```

## Performance

### Expected Results
- **Training Accuracy**: ~85-90%
- **Validation Accuracy**: ~75-80%
- **Training Time**: ~15-20 minutes (30 epochs, GPU)
- **Model Size**: ~50-60 MB

### Optimization Tips
1. **Use GPU**: Install `tensorflow-gpu` for faster training
2. **Increase Batch Size**: If you have enough memory
3. **Data Augmentation**: Already enabled for better generalization
4. **Learning Rate**: Use ReduceLROnPlateau callback (already configured)

## Troubleshooting

### Common Issues

**1. TensorFlow Installation Issues**
```bash
# For M1/M2 Macs
pip install tensorflow-macos tensorflow-metal

# For other systems
pip install tensorflow==2.15.0
```

**2. Port Already in Use**
```bash
# Change port in backend/api/app.py
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

**3. CORS Errors**
- Ensure Flask-CORS is installed
- Check API_BASE_URL in frontend configuration

**4. Model Not Found**
- Train a model first using `python backend/train.py`
- Check `results/` directory for `.h5` files

## API Endpoints

### Dataset
- `GET /api/health` - Health check
- `GET /api/dataset/info` - Dataset information
- `GET /api/dataset/samples` - Get sample images

### Training
- `POST /api/training/start` - Start training
- `GET /api/training/status` - Get training status
- `GET /api/training/progress` - Get training progress
- `GET /api/training/summary` - Get training summary

### Models
- `GET /api/models/list` - List trained models
- `GET /api/models/download/<filename>` - Download model

### Predictions
- `POST /api/predict` - Make prediction on uploaded image

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **CIFAR-10 Dataset**: Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **TensorFlow**: Google Brain Team
- **Astro**: The Astro Technology Company
- **Chart.js**: Chart.js Contributors

## Contact

For questions or support, please open an issue on GitHub.