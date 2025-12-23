"""
Improved training script for CIFAR-10 image classification
Uses modular components for better code organization
"""
from model import CIFAR10Model
from data_loader import CIFAR10DataLoader
from trainer import ModelTrainer


def main():
    """Main training function"""
    
    # Configuration
    config = {
        "batch_size": 64,
        "epochs": 30,
        "learning_rate": 0.001,
        "use_augmentation": True,
        "model_name": "cifar10-cnn-v2",
        "log_dir": "logs",
        "model_dir": "results"
    }
    
    print("="*60)
    print("CIFAR-10 Image Classification Training")
    print("="*60)
    
    # Load data
    print("\n[1/4] Loading and preprocessing data...")
    data_loader = CIFAR10DataLoader(batch_size=config["batch_size"])
    ds_train, ds_test, info = data_loader.load_data(
        use_augmentation=config["use_augmentation"]
    )
    print(f"✓ Training samples: {data_loader.num_train_examples}")
    print(f"✓ Test samples: {data_loader.num_test_examples}")
    print(f"✓ Image shape: {data_loader.input_shape}")
    print(f"✓ Data augmentation: {'enabled' if config['use_augmentation'] else 'disabled'}")
    
    # Build model
    print("\n[2/4] Building model architecture...")
    model = CIFAR10Model(num_classes=10)
    model.build(input_shape=data_loader.input_shape)
    model.compile(
        optimizer="adam",
        learning_rate=config["learning_rate"]
    )
    print("✓ Model built successfully")
    model.summary()
    
    # Setup trainer
    print("\n[3/4] Setting up trainer...")
    trainer = ModelTrainer(model, data_loader, config)
    print("✓ Trainer configured")
    print(f"  - Callbacks: TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau")
    print(f"  - Progress logging enabled for web UI")
    
    # Train model
    print("\n[4/4] Training model...")
    history = trainer.train()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"  Val Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"  Train Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Val Loss: {history.history['val_loss'][-1]:.4f}")
    print(f"\nBest Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"\nModels saved in: {config['model_dir']}/")
    print(f"  - {config['model_name']}-best.h5 (best validation accuracy)")
    print(f"  - {config['model_name']}-final.h5 (final epoch)")
    print(f"\nLogs saved in: {config['log_dir']}/")
    print(f"Training summary: {config['model_dir']}/training_summary.json")
    print("\nTo view training progress, run: tensorboard --logdir={config['log_dir']}")
    print("Or use the web UI: python app.py")
    

if __name__ == "__main__":
    main()