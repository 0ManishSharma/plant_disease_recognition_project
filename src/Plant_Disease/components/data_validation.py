import os
from dataclasses import dataclass


@dataclass
class DataValidationConfig:
    dataset_dir: str = "archive/PlantVillage"


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_dataset(self):
        print("🔍 Starting Data Validation")

        if not os.path.exists(self.config.dataset_dir):
            raise FileNotFoundError("❌ Dataset directory not found")

        class_folders = os.listdir(self.config.dataset_dir)

        if len(class_folders) == 0:
            raise ValueError("❌ No class folders found")

        for cls in class_folders:
            class_path = os.path.join(self.config.dataset_dir, cls)

            if not os.path.isdir(class_path):
                raise ValueError(f"❌ {cls} is not a directory")

            images = os.listdir(class_path)

            if len(images) == 0:
                raise ValueError(f"❌ No images in class {cls}")

        print("✅ Data Validation Successful")
        return True
