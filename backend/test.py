"""
Improved testing script for CIFAR-10 model
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import CIFAR10DataLoader
import os


def evaluate_model(model_path, data_loader):
    """Evaluate model on test set"""
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    
    print("Evaluating model on test set...")
    steps = data_loader.num_test_examples // data_loader.batch_size
    loss, accuracy = model.evaluate(data_loader.ds_test, steps=steps, verbose=1)
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Loss: {loss:.4f}")
    print(f"{'='*60}\n")
    
    return model, loss, accuracy


def visualize_predictions(model, data_loader, num_samples=25):
    """Visualize model predictions on sample images"""
    print(f"Generating predictions for {num_samples} samples...")
    
    # Get sample images
    images, labels = data_loader.get_sample_images(num_samples=num_samples)
    
    # Make predictions
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create visualization
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle('CIFAR-10 Model Predictions', fontsize=16, fontweight='bold')
    
    for idx in range(num_samples):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col] if grid_size > 1 else axes[idx]
        
        # Display image
        ax.imshow(images[idx])
        ax.axis('off')
        
        # Get labels
        true_label = data_loader.get_class_name(labels[idx])
        pred_label = data_loader.get_class_name(predicted_labels[idx])
        confidence = predictions[idx][predicted_labels[idx]] * 100
        
        # Color code: green for correct, red for incorrect
        color = 'green' if true_label == pred_label else 'red'
        
        # Set title
        title = f"True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)"
        ax.set_title(title, fontsize=8, color=color, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(num_samples, grid_size * grid_size):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col] if grid_size > 1 else axes[idx]
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    output_path = 'results/predictions_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    plt.show()
    
    # Calculate accuracy for these samples
    correct = np.sum(predicted_labels == labels)
    sample_accuracy = (correct / num_samples) * 100
    print(f"\nSample Accuracy: {correct}/{num_samples} ({sample_accuracy:.1f}%)")


def test_single_prediction(model, data_loader):
    """Test prediction on a single random image"""
    print("\nTesting single prediction...")
    
    # Get a single sample
    images, labels = data_loader.get_sample_images(num_samples=1)
    image = images[0]
    true_label = labels[0]
    
    # Make prediction
    prediction = model.predict(image.reshape(1, *image.shape), verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class] * 100
    
    # Get class names
    true_class_name = data_loader.get_class_name(true_label)
    pred_class_name = data_loader.get_class_name(predicted_class)
    
    # Display results
    print(f"\nPrediction Results:")
    print(f"  True Label: {true_class_name}")
    print(f"  Predicted: {pred_class_name}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"  Correct: {'✓' if true_class_name == pred_class_name else '✗'}")
    
    # Show all class probabilities
    print(f"\nAll Class Probabilities:")
    for i, prob in enumerate(prediction[0]):
        class_name = data_loader.get_class_name(i)
        bar = '█' * int(prob * 50)
        print(f"  {class_name:12s}: {bar} {prob*100:.2f}%")
    
    # Display image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    color = 'green' if true_class_name == pred_class_name else 'red'
    plt.title(f"True: {true_class_name}\nPredicted: {pred_class_name} ({confidence:.1f}%)",
              fontsize=14, color=color, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Main testing function"""
    print("="*60)
    print("CIFAR-10 Model Testing")
    print("="*60)
    
    # Configuration
    model_path = "results/cifar10-cnn-v2-best.h5"
    batch_size = 64
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at {model_path}")
        print("Please train the model first by running: python train.py")
        
        # Try alternative model path
        alt_path = "results/cifar10-model-v1.h5"
        if os.path.exists(alt_path):
            print(f"\nFound alternative model at {alt_path}")
            model_path = alt_path
        else:
            return
    
    # Load data
    print("\n[1/3] Loading test data...")
    data_loader = CIFAR10DataLoader(batch_size=batch_size)
    data_loader.load_data(use_augmentation=False)
    print(f"✓ Test samples: {data_loader.num_test_examples}")
    
    # Evaluate model
    print("\n[2/3] Evaluating model...")
    model, loss, accuracy = evaluate_model(model_path, data_loader)
    
    # Visualize predictions
    print("\n[3/3] Visualizing predictions...")
    visualize_predictions(model, data_loader, num_samples=25)
    
    # Test single prediction
    test_single_prediction(model, data_loader)
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()