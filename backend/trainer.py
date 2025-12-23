"""
Training utilities and callbacks for CIFAR-10 model
"""
import os
import json
import time
from datetime import datetime
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, Callback
)


class TrainingProgressCallback(Callback):
    """Custom callback to save training progress to JSON for web UI"""
    
    def __init__(self, log_file="training_progress.json"):
        super().__init__()
        self.log_file = log_file
        self.history = {
            "epochs": [],
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
            "timestamp": []
        }
        
    def on_epoch_end(self, epoch, logs=None):
        """Save metrics after each epoch"""
        logs = logs or {}
        
        self.history["epochs"].append(epoch + 1)
        self.history["loss"].append(float(logs.get("loss", 0)))
        self.history["accuracy"].append(float(logs.get("accuracy", 0)))
        self.history["val_loss"].append(float(logs.get("val_loss", 0)))
        self.history["val_accuracy"].append(float(logs.get("val_accuracy", 0)))
        self.history["learning_rate"].append(float(self.model.optimizer.lr.numpy()))
        self.history["timestamp"].append(datetime.now().isoformat())
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)


class ModelTrainer:
    """Handles model training with callbacks and monitoring"""
    
    def __init__(self, model, data_loader, config=None):
        """
        Initialize trainer
        
        Args:
            model: CIFAR10Model instance
            data_loader: CIFAR10DataLoader instance
            config: Training configuration dict
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config or {}
        self.history = None
        
        # Default configuration
        self.epochs = self.config.get("epochs", 30)
        self.log_dir = self.config.get("log_dir", "logs")
        self.model_dir = self.config.get("model_dir", "results")
        self.model_name = self.config.get("model_name", "cifar10-model")
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
    def get_callbacks(self):
        """Setup training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(self.log_dir, f"{self.model_name}-{timestamp}"),
                histogram_freq=1,
                write_graph=True
            ),
            
            # Save best model
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f"{self.model_name}-best.h5"),
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Custom progress callback for web UI
            TrainingProgressCallback(
                log_file=os.path.join(self.model_dir, "training_progress.json")
            )
        ]
        
        return callbacks
    
    def train(self):
        """Train the model"""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.model_name}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.data_loader.batch_size}")
        print(f"Training samples: {self.data_loader.num_train_examples}")
        print(f"Test samples: {self.data_loader.num_test_examples}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Calculate steps
        steps_per_epoch = self.data_loader.num_train_examples // self.data_loader.batch_size
        validation_steps = self.data_loader.num_test_examples // self.data_loader.batch_size
        
        # Train model
        self.history = self.model.model.fit(
            self.data_loader.ds_train,
            epochs=self.epochs,
            validation_data=self.data_loader.ds_test,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"{'='*60}\n")
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, f"{self.model_name}-final.h5")
        self.model.save(final_model_path)
        
        # Save training summary
        self.save_training_summary(training_time)
        
        return self.history
    
    def save_training_summary(self, training_time):
        """Save training summary to JSON"""
        if self.history is None:
            return
            
        summary = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "training_time_seconds": training_time,
            "epochs_completed": len(self.history.history["loss"]),
            "final_metrics": {
                "train_loss": float(self.history.history["loss"][-1]),
                "train_accuracy": float(self.history.history["accuracy"][-1]),
                "val_loss": float(self.history.history["val_loss"][-1]),
                "val_accuracy": float(self.history.history["val_accuracy"][-1])
            },
            "best_metrics": {
                "best_val_accuracy": float(max(self.history.history["val_accuracy"])),
                "best_val_loss": float(min(self.history.history["val_loss"]))
            },
            "config": {
                "epochs": self.epochs,
                "batch_size": self.data_loader.batch_size,
                "num_train_samples": self.data_loader.num_train_examples,
                "num_test_samples": self.data_loader.num_test_examples
            }
        }
        
        summary_path = os.path.join(self.model_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to {summary_path}")
