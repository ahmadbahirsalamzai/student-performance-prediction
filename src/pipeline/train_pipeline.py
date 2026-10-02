from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    best_model_name, score = trainer.initiate_model_trainer(train_arr, test_arr)

    print()
    print("Best model: {} (test R2 = {:.4f})".format(best_model_name, score))


if __name__ == "__main__":
    main()
