import tensorflow as tf
from dataclasses import dataclass
import os
import json


@dataclass
class ModelTrainerConfig:
    image_size: tuple = (224, 224)
    epochs: int = 5
    learning_rate: float = 0.001
    artifacts_dir: str = "artifacts"


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        os.makedirs(self.config.artifacts_dir, exist_ok=True)

    def build_model(self, num_classes: int):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(*self.config.image_size, 3)),

            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])

        return model

    def train(self, train_ds, val_ds, class_names):
        print("🚀 Model Training Started")

        num_classes = len(class_names)

        model = self.build_model(num_classes)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs
        )

        # Save model
        model_path = os.path.join(self.config.artifacts_dir, "model.h5")
        model.save(model_path)

        # Save class names
        class_path = os.path.join(self.config.artifacts_dir, "class_names.json")
        with open(class_path, "w") as f:
            json.dump(class_names, f)

        print("✅ Model & class names saved")
        return model
