"""
Data loading and preprocessing utilities for Training Data - Image Clasifier
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class VisionDataLoader:
    """General purpose dataset loader with advanced AI preprocessing"""
    
    def __init__(self, batch_size=32, target_size=(224, 224)):
        """
        Initialize loader
        
        Args:
            batch_size: Training batch size
            target_size: Resize images to this size (required for most AI models)
        """
        self.batch_size = batch_size
        self.target_size = target_size
        self.info = None
        self.ds_train = None
        self.ds_test = None
        
    def preprocess_image(self, image, label):
        """
        Resize and normalize image for AI models.
        Uses MobileNetV2 style normalization [-1, 1].
        """
        image = tf.image.resize(image, self.target_size)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label
    
    def augment_image(self, image, label):
        """Apply advanced data augmentation to boost AI robustness"""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        # Random rotation simulation via small shift
        image = tf.image.random_jpeg_quality(image, min_jpeg_quality=80, max_jpeg_quality=100)
        return image, label
    
    def load_dataset(self, name="cifar10", use_augmentation=True):
        """Load any dataset from TFDS with AI preprocessing"""
        ds_train, self.info = tfds.load(
            name,
            with_info=True,
            split="train",
            as_supervised=True
        )
        ds_test = tfds.load(name, split="test", as_supervised=True)
        
        # Training Pipeline
        ds_train = ds_train.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        if use_augmentation:
            ds_train = ds_train.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        ds_train = ds_train.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Test Pipeline
        ds_test = ds_test.map(self.preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        self.ds_train = ds_train
        self.ds_test = ds_test
        return ds_train, ds_test
    
    @property
    def input_shape(self):
        return (*self.target_size, 3)


class CIFAR10DataLoader(VisionDataLoader):
    """Compatibility loader for CIFAR-10"""
    
    CATEGORIES = {
        0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
        5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
    }

    def __init__(self, batch_size=32):
        # CIFAR-10 is naturally 32x32, but we resize to 224x224 for "Power AI"
        super().__init__(batch_size=batch_size, target_size=(224, 224))

    def load_data(self, use_augmentation=True):
        return self.load_dataset("cifar10", use_augmentation=use_augmentation)

    def get_class_name(self, label):
        return self.CATEGORIES.get(int(label), "unknown")

    def get_sample_images(self, num_samples=25, from_test=True):
        dataset = self.ds_test if from_test else self.ds_train
        images, labels = [], []
        for img_batch, lab_batch in dataset.take(1):
            for i in range(min(num_samples, len(img_batch))):
                # Denormalize for visualization
                img = (img_batch[i].numpy() + 1.0) * 127.5
                images.append(img.astype(np.uint8))
                labels.append(lab_batch[i].numpy())
        return np.array(images), np.array(labels)

    @property
    def num_train_examples(self):
        return 50000 # Standard CIFAR10

    @property
    def num_test_examples(self):
        return 10000
