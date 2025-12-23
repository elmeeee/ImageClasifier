# 📋 Quick Reference Guide

## 🚀 Quick Start Commands

### Setup (First Time Only)
```bash
./setup.sh
```

### Development

**Start Backend API:**
```bash
cd backend/api
python app.py
```

**Start Frontend (New Terminal):**
```bash
cd frontend
npm run dev
```

**Access Dashboard:**
```
http://localhost:4321
```

### Training

**Train Model (CLI):**
```bash
cd backend
python train.py
```

**Test Model:**
```bash
cd backend
python test.py
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `backend/model.py` | CNN architecture |
| `backend/data_loader.py` | Dataset loading |
| `backend/trainer.py` | Training utilities |
| `backend/train.py` | Training script |
| `backend/test.py` | Testing script |
| `backend/api/app.py` | Flask API server |
| `frontend/src/pages/index.astro` | Main dashboard page |
| `frontend/src/scripts/main.js` | Frontend logic |
| `vercel.json` | Deployment config |

## 🎯 Common Tasks

### Change Training Parameters
Edit `backend/train.py`:
```python
config = {
    "batch_size": 64,      # ← Change this
    "epochs": 30,          # ← Change this
    "learning_rate": 0.001 # ← Change this
}
```

### Add New Model Architecture
Edit `backend/model.py` → `build()` method

### Customize UI Theme
Edit `frontend/src/layouts/Layout.astro` → `:root` CSS variables

### Change API Port
Edit `backend/api/app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # ← Change port
```

## 🔧 Troubleshooting

### "Module not found" Error
```bash
pip install -r requirements.txt
```

### "Port already in use"
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9
```

### Frontend not connecting to API
Check `frontend/.env`:
```env
PUBLIC_API_URL=http://localhost:5000/api
```

### TensorFlow GPU not working
```bash
# For M1/M2 Macs
pip install tensorflow-macos tensorflow-metal

# For NVIDIA GPUs
pip install tensorflow-gpu
```

## 📊 Model Files

### Trained Models Location
```
results/
├── cifar10-cnn-v2-best.h5    # Best model (highest val accuracy)
├── cifar10-cnn-v2-final.h5   # Final epoch model
├── training_progress.json     # Training metrics
└── training_summary.json      # Training summary
```

### TensorBoard Logs
```
logs/
└── cifar10-cnn-v2-YYYYMMDD-HHMMSS/
```

**View TensorBoard:**
```bash
tensorboard --logdir=logs
```

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/dataset/info` | Dataset info |
| GET | `/api/dataset/samples` | Sample images |
| POST | `/api/training/start` | Start training |
| GET | `/api/training/status` | Training status |
| GET | `/api/training/progress` | Training progress |
| GET | `/api/training/summary` | Training summary |
| GET | `/api/models/list` | List models |
| GET | `/api/models/download/<name>` | Download model |
| POST | `/api/predict` | Make prediction |

## 🎨 UI Components

### Dataset Explorer
- Shows dataset statistics
- Displays sample images
- Interactive category badges

### Training Panel
- Configure hyperparameters
- Start training
- View status

### Progress Monitor
- Real-time charts
- Training metrics
- Summary statistics

### Models Manager
- List trained models
- Download models
- Model metadata

### Prediction Panel
- Upload images
- Select model
- View predictions

## 📦 Dependencies

### Python
- tensorflow==2.11.1
- flask
- flask-cors
- pillow
- numpy
- matplotlib
- tensorflow_datasets

### Node.js
- astro
- @astrojs/vercel
- chart.js

## 🚀 Deployment

### Vercel (Recommended)
```bash
vercel
```

### Build Frontend Only
```bash
cd frontend
npm run build
```

### Build for Production
```bash
# Frontend
cd frontend
npm run vercel-build

# Backend (if deploying separately)
# No build needed, just ensure requirements.txt is up to date
```

## 📈 Performance Tips

1. **Use GPU** for training (10x faster)
2. **Increase batch size** if you have memory
3. **Enable data augmentation** (already on by default)
4. **Use early stopping** (already configured)
5. **Monitor with TensorBoard** for insights

## 🔐 Security Notes

- Never commit `.env` files
- Keep API keys in environment variables
- Use CORS properly in production
- Implement rate limiting for public APIs

## 📞 Getting Help

1. Check `README.md` for detailed docs
2. Check `DEPLOYMENT.md` for Vercel deployment
3. Open an issue on GitHub
4. Check TensorFlow/Astro documentation

## 🎯 Project Goals

✅ Modern, beautiful UI
✅ Real-time training monitoring  
✅ Easy model management
✅ Production-ready deployment
✅ Clean, modular code
✅ Comprehensive documentation

---

**Happy Coding! 🎉**
