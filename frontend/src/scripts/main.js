// CIFAR-10 Training Dashboard - Main JavaScript
// API Configuration
const API_BASE_URL = import.meta.env.PUBLIC_API_URL || '/api';

// Global state
let charts = {
    accuracy: null,
    loss: null,
    learningRate: null
};

let progressInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    initializeTabs();
    loadDatasetInfo();
    loadSamples();
    loadModels();
    initializeUpload();
    initializeTraining();

    // Start monitoring training progress
    startProgressMonitoring();
});

// Tab Navigation
function initializeTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;

            // Update active states
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(tabName).classList.add('active');

            // Load data for specific tabs
            if (tabName === 'progress') {
                loadTrainingProgress();
            } else if (tabName === 'models') {
                loadModels();
            }
        });
    });
}

// Dataset Info
async function loadDatasetInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/dataset/info`);
        const data = await response.json();

        document.getElementById('trainSamples').textContent = data.num_train_samples.toLocaleString();
        document.getElementById('testSamples').textContent = data.num_test_samples.toLocaleString();
        document.getElementById('numClasses').textContent = data.num_classes;
        document.getElementById('imageShape').textContent = data.input_shape.join(' × ');

        // Display categories
        const categoriesGrid = document.getElementById('categoriesGrid');
        categoriesGrid.innerHTML = '';

        Object.entries(data.categories).forEach(([id, name]) => {
            const badge = document.createElement('div');
            badge.className = 'category-badge';
            badge.textContent = name;
            badge.style.cssText = `
                background: var(--bg-card);
                backdrop-filter: blur(10px);
                border: 1px solid var(--border-color);
                padding: 1rem;
                border-radius: 0.75rem;
                text-align: center;
                font-weight: 600;
                transition: var(--transition);
                cursor: pointer;
            `;
            badge.addEventListener('mouseenter', function () {
                this.style.background = 'var(--gradient-primary)';
                this.style.transform = 'scale(1.05)';
                this.style.boxShadow = 'var(--shadow-lg)';
            });
            badge.addEventListener('mouseleave', function () {
                this.style.background = 'var(--bg-card)';
                this.style.transform = 'scale(1)';
                this.style.boxShadow = 'none';
            });
            categoriesGrid.appendChild(badge);
        });

    } catch (error) {
        console.error('Error loading dataset info:', error);
    }
}

// Load Sample Images
async function loadSamples() {
    const samplesGrid = document.getElementById('samplesGrid');
    samplesGrid.innerHTML = '<div class="loading">Loading samples...</div>';

    const fromTest = document.querySelector('input[name="sampleSource"]:checked').value === 'test';

    try {
        const response = await fetch(`${API_BASE_URL}/dataset/samples?num_samples=25&from_test=${fromTest}`);
        const data = await response.json();

        samplesGrid.innerHTML = '';

        data.samples.forEach(sample => {
            const item = document.createElement('div');
            item.className = 'sample-item';
            item.style.cssText = `
                background: var(--bg-card);
                backdrop-filter: blur(10px);
                border: 1px solid var(--border-color);
                border-radius: 0.75rem;
                padding: 0.75rem;
                transition: var(--transition);
                overflow: hidden;
            `;
            item.innerHTML = `
                <img src="${sample.image}" alt="${sample.label_name}" style="width: 100%; height: auto; border-radius: 0.5rem; margin-bottom: 0.5rem;">
                <div style="font-size: 0.85rem; font-weight: 600; text-align: center; color: var(--text-secondary);">${sample.label_name}</div>
            `;
            item.addEventListener('mouseenter', function () {
                this.style.transform = 'scale(1.05)';
                this.style.boxShadow = 'var(--shadow-xl)';
                this.style.borderColor = 'var(--primary)';
            });
            item.addEventListener('mouseleave', function () {
                this.style.transform = 'scale(1)';
                this.style.boxShadow = 'none';
                this.style.borderColor = 'var(--border-color)';
            });
            samplesGrid.appendChild(item);
        });

    } catch (error) {
        console.error('Error loading samples:', error);
        samplesGrid.innerHTML = '<div class="loading">Error loading samples</div>';
    }
}

// Refresh samples button
document.addEventListener('DOMContentLoaded', () => {
    const refreshBtn = document.getElementById('refreshSamples');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadSamples);
    }
});

// Training
function initializeTraining() {
    const startBtn = document.getElementById('startTrainingBtn');
    if (startBtn) {
        startBtn.addEventListener('click', startTraining);
    }
}

async function startTraining() {
    const btn = document.getElementById('startTrainingBtn');
    const statusBox = document.getElementById('trainingStatusBox');
    const statusMessage = document.getElementById('trainingMessage');

    const config = {
        epochs: parseInt(document.getElementById('epochs').value),
        batch_size: parseInt(document.getElementById('batchSize').value),
        learning_rate: parseFloat(document.getElementById('learningRate').value)
    };

    btn.disabled = true;
    btn.innerHTML = '<span class="btn-icon">⏳</span> Starting Training...';

    try {
        const response = await fetch(`${API_BASE_URL}/training/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (response.ok) {
            statusBox.style.display = 'flex';
            statusMessage.textContent = 'Training started! Check the Progress tab for real-time updates.';

            // Switch to progress tab
            setTimeout(() => {
                document.querySelector('[data-tab="progress"]').click();
            }, 2000);
        } else {
            statusMessage.textContent = `Error: ${data.error}`;
            statusBox.style.display = 'flex';
            btn.disabled = false;
            btn.innerHTML = '<span class="btn-icon">🚀</span> Start Training';
        }

    } catch (error) {
        console.error('Error starting training:', error);
        statusMessage.textContent = 'Error starting training. Please try again.';
        statusBox.style.display = 'flex';
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🚀</span> Start Training';
    }
}

// Training Progress Monitoring
function startProgressMonitoring() {
    // Check status every 5 seconds
    progressInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/training/status`);
            const status = await response.json();

            const statusElement = document.getElementById('trainingStatus');
            if (statusElement) {
                if (status.is_training) {
                    statusElement.textContent = 'Training...';
                    statusElement.style.color = 'var(--warning)';

                    // Auto-refresh progress charts if on progress tab
                    if (document.getElementById('progress').classList.contains('active')) {
                        loadTrainingProgress();
                    }
                } else {
                    statusElement.textContent = 'Ready';
                    statusElement.style.color = 'var(--success)';
                }
            }

        } catch (error) {
            console.error('Error checking training status:', error);
        }
    }, 5000);
}

// Load Training Progress
async function loadTrainingProgress() {
    try {
        const response = await fetch(`${API_BASE_URL}/training/progress`);
        const data = await response.json();

        if (data.epochs && data.epochs.length > 0) {
            updateCharts(data);
            loadTrainingSummary();
        }

    } catch (error) {
        console.error('Error loading training progress:', error);
    }
}

// Update Charts
function updateCharts(data) {
    // Import Chart.js dynamically
    if (typeof Chart === 'undefined') {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js';
        script.onload = () => updateCharts(data);
        document.head.appendChild(script);
        return;
    }

    const chartConfig = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                labels: {
                    color: '#cbd5e1'
                }
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(148, 163, 184, 0.1)'
                },
                ticks: {
                    color: '#94a3b8'
                }
            },
            y: {
                grid: {
                    color: 'rgba(148, 163, 184, 0.1)'
                },
                ticks: {
                    color: '#94a3b8'
                }
            }
        }
    };

    // Accuracy Chart
    const accCtx = document.getElementById('accuracyChart');
    if (accCtx) {
        if (charts.accuracy) {
            charts.accuracy.destroy();
        }
        charts.accuracy = new Chart(accCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: data.epochs,
                datasets: [
                    {
                        label: 'Training Accuracy',
                        data: data.accuracy,
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Validation Accuracy',
                        data: data.val_accuracy,
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: chartConfig
        });
    }

    // Loss Chart
    const lossCtx = document.getElementById('lossChart');
    if (lossCtx) {
        if (charts.loss) {
            charts.loss.destroy();
        }
        charts.loss = new Chart(lossCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: data.epochs,
                datasets: [
                    {
                        label: 'Training Loss',
                        data: data.loss,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Validation Loss',
                        data: data.val_loss,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: chartConfig
        });
    }

    // Learning Rate Chart
    const lrCtx = document.getElementById('learningRateChart');
    if (lrCtx) {
        if (charts.learningRate) {
            charts.learningRate.destroy();
        }
        charts.learningRate = new Chart(lrCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: data.epochs,
                datasets: [
                    {
                        label: 'Learning Rate',
                        data: data.learning_rate,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: chartConfig
        });
    }
}

// Load Training Summary
async function loadTrainingSummary() {
    try {
        const response = await fetch(`${API_BASE_URL}/training/summary`);
        const data = await response.json();

        if (data.error) return;

        const summarySection = document.getElementById('summarySection');
        const summaryGrid = document.getElementById('summaryGrid');

        if (summaryGrid) {
            summaryGrid.innerHTML = `
                <div class="summary-item" style="padding: 1rem; background: var(--bg-secondary); border-radius: 0.5rem;">
                    <h4 style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Training Time</h4>
                    <p style="font-size: 1.5rem; font-weight: 700; color: var(--primary-light);">${(data.training_time_seconds / 60).toFixed(1)} min</p>
                </div>
                <div class="summary-item" style="padding: 1rem; background: var(--bg-secondary); border-radius: 0.5rem;">
                    <h4 style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Epochs Completed</h4>
                    <p style="font-size: 1.5rem; font-weight: 700; color: var(--primary-light);">${data.epochs_completed}</p>
                </div>
                <div class="summary-item" style="padding: 1rem; background: var(--bg-secondary); border-radius: 0.5rem;">
                    <h4 style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Final Train Accuracy</h4>
                    <p style="font-size: 1.5rem; font-weight: 700; color: var(--primary-light);">${(data.final_metrics.train_accuracy * 100).toFixed(2)}%</p>
                </div>
                <div class="summary-item" style="padding: 1rem; background: var(--bg-secondary); border-radius: 0.5rem;">
                    <h4 style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Final Val Accuracy</h4>
                    <p style="font-size: 1.5rem; font-weight: 700; color: var(--primary-light);">${(data.final_metrics.val_accuracy * 100).toFixed(2)}%</p>
                </div>
                <div class="summary-item" style="padding: 1rem; background: var(--bg-secondary); border-radius: 0.5rem;">
                    <h4 style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Best Val Accuracy</h4>
                    <p style="font-size: 1.5rem; font-weight: 700; color: var(--primary-light);">${(data.best_metrics.best_val_accuracy * 100).toFixed(2)}%</p>
                </div>
                <div class="summary-item" style="padding: 1rem; background: var(--bg-secondary); border-radius: 0.5rem;">
                    <h4 style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">Best Val Loss</h4>
                    <p style="font-size: 1.5rem; font-weight: 700; color: var(--primary-light);">${data.best_metrics.best_val_loss.toFixed(4)}</p>
                </div>
            `;

            if (summarySection) {
                summarySection.style.display = 'block';
            }
        }

    } catch (error) {
        console.error('Error loading training summary:', error);
    }
}

// Load Models
async function loadModels() {
    const modelsList = document.getElementById('modelsList');
    if (!modelsList) return;

    modelsList.innerHTML = '<div class="loading">Loading models...</div>';

    try {
        const response = await fetch(`${API_BASE_URL}/models/list`);
        const data = await response.json();

        if (data.models.length === 0) {
            modelsList.innerHTML = '<div class="loading">No trained models found. Train a model first!</div>';
            return;
        }

        modelsList.innerHTML = '';

        data.models.forEach(model => {
            const item = document.createElement('div');
            item.className = 'model-item';

            const date = new Date(model.modified * 1000);
            const size = (model.size / (1024 * 1024)).toFixed(2);

            item.style.cssText = `
                background: var(--bg-card);
                backdrop-filter: blur(10px);
                border: 1px solid var(--border-color);
                border-radius: 1rem;
                padding: 1.5rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                transition: var(--transition);
            `;

            item.innerHTML = `
                <div class="model-info">
                    <h4 style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem;">📦 ${model.name}</h4>
                    <div style="display: flex; gap: 1.5rem; color: var(--text-muted); font-size: 0.9rem;">
                        <span>💾 ${size} MB</span>
                        <span>📅 ${date.toLocaleDateString()} ${date.toLocaleTimeString()}</span>
                    </div>
                </div>
                <button class="btn btn-primary" onclick="downloadModel('${model.name}')" style="padding: 0.75rem 1.5rem; border: none; border-radius: 0.5rem; font-weight: 600; cursor: pointer; background: var(--gradient-primary); color: white; box-shadow: var(--shadow-md);">
                    ⬇️ Download
                </button>
            `;

            item.addEventListener('mouseenter', function () {
                this.style.borderColor = 'var(--primary)';
                this.style.boxShadow = 'var(--shadow-lg)';
            });
            item.addEventListener('mouseleave', function () {
                this.style.borderColor = 'var(--border-color)';
                this.style.boxShadow = 'none';
            });

            modelsList.appendChild(item);
        });

        // Update model selector in predict tab
        updateModelSelector(data.models);

    } catch (error) {
        console.error('Error loading models:', error);
        modelsList.innerHTML = '<div class="loading">Error loading models</div>';
    }
}

// Update Model Selector
function updateModelSelector(models) {
    const select = document.getElementById('modelSelect');
    if (!select) return;

    select.innerHTML = '';

    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.name;
        option.textContent = model.name;
        select.appendChild(option);
    });
}

// Download Model
window.downloadModel = function (filename) {
    window.location.href = `${API_BASE_URL}/models/download/${filename}`;
}

// Image Upload
function initializeUpload() {
    const uploadBox = document.getElementById('uploadBox');
    const fileInput = document.getElementById('imageUpload');

    if (!uploadBox || !fileInput) return;

    uploadBox.addEventListener('click', () => {
        fileInput.click();
    });

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--primary)';
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = 'var(--border-color)';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--border-color)';

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageUpload(file);
        }
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImageUpload(file);
        }
    });
}

// Handle Image Upload
async function handleImageUpload(file) {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('model', document.getElementById('modelSelect').value);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayPrediction(file, data);
        } else {
            alert(`Error: ${data.error}`);
        }

    } catch (error) {
        console.error('Error making prediction:', error);
        alert('Error making prediction. Please try again.');
    }
}

// Display Prediction
function displayPrediction(file, data) {
    const resultsDiv = document.getElementById('predictionResults');
    const uploadedImage = document.getElementById('uploadedImage');
    const predictedLabel = document.getElementById('predictedLabel');
    const confidence = document.getElementById('confidence');
    const probabilities = document.getElementById('probabilities');

    if (!resultsDiv) return;

    // Display image
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImage.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Display prediction
    predictedLabel.textContent = data.predicted_label;
    confidence.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

    // Display all probabilities
    probabilities.innerHTML = '';

    // Sort probabilities
    const sortedProbs = Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1]);

    sortedProbs.forEach(([label, prob]) => {
        const item = document.createElement('div');
        item.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: var(--bg-secondary);
            border-radius: 0.5rem;
        `;
        item.innerHTML = `
            <span>${label}</span>
            <div style="flex: 1; height: 8px; background: var(--bg-tertiary); border-radius: 4px; margin: 0 1rem; overflow: hidden;">
                <div style="height: 100%; background: var(--gradient-primary); border-radius: 4px; width: ${prob * 100}%; transition: width 0.5s ease-out;"></div>
            </div>
            <span>${(prob * 100).toFixed(1)}%</span>
        `;
        probabilities.appendChild(item);
    });

    resultsDiv.style.display = 'grid';
}
