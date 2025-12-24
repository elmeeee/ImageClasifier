"""
Model architecture for image classification using Transfer Learning
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten,
    GlobalAveragePooling2D, BatchNormalization, Input
)
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0


class VisionAIModel:
    """
    Powerful Image Classification Model using Transfer Learning.
    Supports MobileNetV2 and EfficientNetB0 as base models.
    """
    
    def __init__(self, num_classes=10, base_model_type="mobilenet_v2"):
        """
        Initialize the model
        
        Args:
            num_classes: Number of output classes
            base_model_type: "mobilenet_v2" or "efficientnet_b0"
        """
        self.num_classes = num_classes
        self.base_model_type = base_model_type
        self.model = None
        self.base_model = None
        
    def build(self, input_shape=(224, 224, 3)):
        """
        Build the model using Transfer Learning:
        1. Pre-trained base model (frozen)
        2. Global Average Pooling
        3. Dense hidden layers with BatchNorm & Dropout
        4. Softmax output layer
        """
        inputs = Input(shape=input_shape)
        
        # Load pre-trained base model
        if self.base_model_type == "mobilenet_v2":
            self.base_model = MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
        else:
            self.base_model = EfficientNetB0(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )
            
        # Freeze the base model to preserve learned features
        self.base_model.trainable = False
        
        # Build the "Power Head"
        x = self.base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        
        # Dense Layer 1
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.4)(x)
        
        # Dense Layer 2
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        # Output Layer
        outputs = Dense(self.num_classes, activation="softmax")(x)
        
        self.model = Model(inputs, outputs)
        return self.model
    
    def compile(self, optimizer="adam", learning_rate=0.0001):
        """
        Compile the model. 
        Uses a lower learning rate by default for transfer learning stability.
        """
        from tensorflow.keras.optimizers import Adam
        opt = Adam(learning_rate=learning_rate) if optimizer == "adam" else optimizer
            
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"]
        )
        
    def unfreeze_base_model(self, fine_tune_at=100):
        """
        Unfreeze the base model for fine-tuning.
        This provides a massive power boost by allowing the AI to 
        adjust its "deep" features to your specific dataset.
        """
        if self.base_model is None:
            return
            
        self.base_model.trainable = True
        
        # Freeze layers before the fine_tune_at layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
            
    def save(self, filepath):
        """Save model to file"""
        if self.model:
            self.model.save(filepath)
            print(f"VisionAI Power Model saved to {filepath}")


class CIFAR10Model(VisionAIModel):
    """Backward compatibility for CIFAR-10 tasks using the new Power Engine"""
    def __init__(self, num_classes=10):
        super().__init__(num_classes=num_classes, base_model_type="mobilenet_v2")
    
    def build(self, input_shape=(32, 32, 3)):
        # For CIFAR-10, we still use MobileNetV2 but with appropriate input shape
        return super().build(input_shape=input_shape)
