from src.Plant_Disease.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.Plant_Disease.components.data_validation import DataValidation, DataValidationConfig
from src.Plant_Disease.components.data_transformation import DataTransformation, DataTransformationConfig
from src.Plant_Disease.components.model_tranier import ModelTrainer,ModelTrainerConfig
# 1️⃣ Validation
validator = DataValidation(DataValidationConfig())
validator.validate_dataset()

# 2️⃣ Ingestion
ingestion = DataIngestion(DataIngestionConfig())
train_ds, val_ds, class_names = ingestion.initiate_data_ingestion()

# 3️⃣ Transformation
transformer = DataTransformation(DataTransformationConfig())
train_ds, val_ds = transformer.transform(train_ds, val_ds)
model_trainer = ModelTrainer(ModelTrainerConfig())
model_trainer.train(train_ds,val_ds,class_names)
print("Model is  ready ✅")
