# 🚀 Vercel Deployment Guide

## Prerequisites
- GitHub account
- Vercel account (sign up at https://vercel.com)
- Git repository with your code

## Method 1: Deploy via Vercel Dashboard (Recommended)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: CIFAR-10 Dashboard"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Import to Vercel
1. Go to https://vercel.com/dashboard
2. Click "Add New..." → "Project"
3. Import your GitHub repository
4. Configure project:
   - **Framework Preset**: Astro
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run vercel-build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

### Step 3: Configure Environment Variables (Optional)
In Vercel project settings → Environment Variables:
- Add any custom environment variables if needed

### Step 4: Deploy
- Click "Deploy"
- Wait for build to complete
- Your app will be live at `https://your-project.vercel.app`

## Method 2: Deploy via Vercel CLI

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Login to Vercel
```bash
vercel login
```

### Step 3: Deploy
```bash
# From project root
vercel

# Follow the prompts:
# - Set up and deploy? Yes
# - Which scope? Select your account
# - Link to existing project? No
# - Project name? (default or custom)
# - In which directory is your code located? ./
```

### Step 4: Production Deployment
```bash
vercel --prod
```

## Important Notes

### 1. API Endpoints
The Python API will be deployed as Vercel Serverless Functions. They will be available at:
```
https://your-project.vercel.app/api/*
```

### 2. File Storage Limitations
⚠️ **Important**: Vercel serverless functions have limitations:
- **Read-only filesystem** (except `/tmp`)
- **500MB deployment size limit**
- **Execution timeout**: 10 seconds (Hobby), 60 seconds (Pro)

**For Training Models:**
- Training models on Vercel is **NOT recommended** due to:
  - Execution time limits
  - No persistent storage
  - Limited compute resources

**Recommended Approach:**
1. Train models locally or on a dedicated server
2. Upload trained models to cloud storage (AWS S3, Google Cloud Storage, etc.)
3. Load models from cloud storage for predictions

### 3. Modified Architecture for Production

For production deployment, consider this architecture:

```
┌─────────────────┐
│  Vercel (Frontend) │
│  - Astro UI      │
│  - Static Assets │
└────────┬────────┘
         │
         ├──────────────────────┐
         │                      │
┌────────▼────────┐    ┌───────▼────────┐
│ Vercel Functions│    │ Cloud Storage  │
│ - Predictions   │    │ - Trained Models│
│ - Dataset Info  │    │ - Training Data │
└─────────────────┘    └────────────────┘
```

### 4. Environment Variables for Production

Create a `.env.production` in frontend:
```env
PUBLIC_API_URL=/api
```

### 5. Optimizing for Vercel

**Update `vercel.json`** for better performance:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "frontend/package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "dist"
      }
    },
    {
      "src": "api/index.py",
      "use": "@vercel/python",
      "config": {
        "maxLambdaSize": "50mb"
      }
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/index.py"
    },
    {
      "handle": "filesystem"
    },
    {
      "src": "/(.*)",
      "dest": "/frontend/$1"
    }
  ],
  "functions": {
    "api/index.py": {
      "memory": 3008,
      "maxDuration": 60
    }
  }
}
```

## Alternative: Hybrid Deployment

### Option 1: Frontend on Vercel + Backend on Railway/Render
1. Deploy frontend to Vercel
2. Deploy backend to Railway (https://railway.app) or Render (https://render.com)
3. Update `PUBLIC_API_URL` to point to your backend URL

### Option 2: Full Stack on Railway
1. Deploy entire project to Railway
2. Railway supports both Node.js and Python
3. Provides persistent storage for models

## Monitoring & Debugging

### View Logs
```bash
vercel logs <deployment-url>
```

### Check Build Status
```bash
vercel inspect <deployment-url>
```

### Rollback Deployment
```bash
vercel rollback
```

## Custom Domain

### Add Custom Domain
1. Go to Vercel Dashboard → Your Project → Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed
4. Wait for DNS propagation

## Troubleshooting

### Build Fails
- Check build logs in Vercel dashboard
- Ensure all dependencies are in `package.json`
- Verify `vercel-build` script works locally

### API Not Working
- Check function logs
- Verify `vercel.json` routes configuration
- Ensure Python dependencies are in `requirements.txt`

### Large Model Files
- Use `.vercelignore` to exclude large files
- Store models in cloud storage
- Load models on-demand

## Cost Considerations

### Vercel Free Tier Limits
- 100 GB bandwidth/month
- 100 hours serverless function execution
- 6,000 minutes build time

For production with heavy usage, consider:
- Vercel Pro plan
- Alternative hosting for backend (Railway, Render, AWS)

## Security Best Practices

1. **Never commit sensitive data**
   - Use environment variables
   - Add `.env` to `.gitignore`

2. **API Rate Limiting**
   - Implement rate limiting for API endpoints
   - Use Vercel Edge Config for rate limits

3. **CORS Configuration**
   - Configure CORS properly in Flask app
   - Restrict origins in production

## Support

For issues:
- Vercel Documentation: https://vercel.com/docs
- Vercel Community: https://github.com/vercel/vercel/discussions
- Project Issues: Open an issue on your GitHub repo

---

**Happy Deploying! 🚀**
