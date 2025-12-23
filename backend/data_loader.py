"""
Data loading and preprocessing utilities for CIFAR-10
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class CIFAR10DataLoader:
    """CIFAR-10 dataset loader with preprocessing"""
    
    CATEGORIES = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }
    
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.info = None
        self.ds_train = None
        self.ds_test = None
        
    def preprocess_image(self, image, label):
        """Convert image from [0, 255] to [0, 1] range"""
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label
    
    def augment_image(self, image, label):
        """Apply data augmentation to training images"""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, label
    
    def load_data(self, use_augmentation=True):
        """
        Load CIFAR-10 dataset with preprocessing
        
        Args:
            use_augmentation: Apply data augmentation to training set
            
        Returns:
            ds_train, ds_test, info
        """
        # Load dataset
        ds_train, self.info = tfds.load(
            "cifar10",
            with_info=True,
            split="train",
            as_supervised=True
        )
        ds_test = tfds.load("cifar10", split="test", as_supervised=True)
        
        # Preprocess training data
        ds_train = ds_train.map(
            self.preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation if requested
        if use_augmentation:
            ds_train = ds_train.map(
                self.augment_image,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Shuffle, batch, and prefetch training data
        ds_train = ds_train.shuffle(10000).batch(self.batch_size).prefetch(
            tf.data.AUTOTUNE
        )
        
        # Preprocess test data
        ds_test = ds_test.map(
            self.preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        self.ds_train = ds_train
        self.ds_test = ds_test
        
        return ds_train, ds_test, self.info
    
    def get_sample_images(self, num_samples=25, from_test=True):
        """
        Get sample images for visualization
        
        Args:
            num_samples: Number of samples to retrieve
            from_test: Use test set (True) or train set (False)
            
        Returns:
            images, labels arrays
        """
        dataset = self.ds_test if from_test else self.ds_train
        
        if dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        images = []
        labels = []
        
        for image_batch, label_batch in dataset.take(1):
            for i in range(min(num_samples, len(image_batch))):
                images.append(image_batch[i].numpy())
                labels.append(label_batch[i].numpy())
                
        return np.array(images), np.array(labels)
    
    def get_class_name(self, label):
        """Convert numeric label to class name"""
        return self.CATEGORIES.get(int(label), "unknown")
    
    @property
    def num_train_examples(self):
        """Get number of training examples"""
        if self.info:
            return self.info.splits["train"].num_examples
        return 0
    
    @property
    def num_test_examples(self):
        """Get number of test examples"""
        if self.info:
            return self.info.splits["test"].num_examples
        return 0
    
    @property
    def input_shape(self):
        """Get input image shape"""
        if self.info:
            return self.info.features["image"].shape
        return (32, 32, 3)
