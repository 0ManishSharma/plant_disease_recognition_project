import tensorflow as tf
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    shuffle_buffer: int = 500


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform(self, train_ds, val_ds):
        print("🔄 Starting Data Transformation")

        # Normalize images
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

        train_ds = train_ds.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=1
        )

        val_ds = val_ds.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=1
        )

        # ⚠️ SAFE PERFORMANCE SETTINGS (NO CRASH)
        train_ds = train_ds.shuffle(self.config.shuffle_buffer).prefetch(1)
        val_ds = val_ds.prefetch(1)

        print("✅ Data Transformation Completed")
        return train_ds, val_ds
