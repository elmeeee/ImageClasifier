"""
Model architecture for CIFAR-10 image classification
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Activation, Flatten,
    Conv2D, MaxPooling2D, BatchNormalization
)


class CIFAR10Model:
    """Enhanced CIFAR-10 CNN Model with BatchNormalization"""
    
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.model = None
        
    def build(self, input_shape=(32, 32, 3)):
        """
        Constructs an improved CNN model with BatchNormalization:
        - Block 1: 2x Conv(32) + BatchNorm + MaxPool + Dropout
        - Block 2: 2x Conv(64) + BatchNorm + MaxPool + Dropout
        - Block 3: 2x Conv(128) + BatchNorm + MaxPool + Dropout
        - Dense: Flatten + FC(1024) + Dropout + FC(num_classes)
        """
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(32, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Dense layers
            Flatten(),
            Dense(1024),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.5),
            Dense(self.num_classes, activation="softmax")
        ])
        
        self.model = model
        return model
    
    def compile(self, optimizer="adam", learning_rate=0.001):
        """Compile the model with specified optimizer"""
        if optimizer == "adam":
            from tensorflow.keras.optimizers import Adam
            opt = Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
            
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"]
        )
        
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Call build() first.")
            
    def save(self, filepath):
        """Save model to file"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Build and train first.")
