import tensorflow as tf
from dataclasses import dataclass
import os


@dataclass
class DataIngestionConfig:
    dataset_dir: str = "archive/PlantVillage"
    image_size: tuple = (224, 224)
    batch_size: int = 8
    validation_split: float = 0.2
    seed: int = 42


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        print("📥 Starting Data Ingestion")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.dataset_dir,
            validation_split=self.config.validation_split,
            subset="training",
            seed=self.config.seed,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.config.dataset_dir,
            validation_split=self.config.validation_split,
            subset="validation",
            seed=self.config.seed,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size
        )

        # ✅ VERY IMPORTANT
        class_names = train_ds.class_names

        print("✅ Data Ingestion Completed")
        print("Classes:", class_names)

        return train_ds, val_ds, class_names
