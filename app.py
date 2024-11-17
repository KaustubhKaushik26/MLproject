from src.mlproject.logger import log as logging
from src.mlproject.exception import CustomException
import sys
from src.mlproject.component.data_ingestion import DataIngestion
from src.mlproject.component.data_ingestion import DataIngestionConfig
from src.mlproject.component.data_tranformation import DataTranformationConfig,DataTranformation
from src.mlproject.component.model_trainer import ModelTrainerConfig,ModelTrainer

if __name__=='__main__':
    logging.info('The execution has started')

    
    try:
        #data_ingestion_config=data_ingestion_config()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
        
        #data_tranformation_config = DataIngestionConfig()
        data_tranformation=DataTranformation()
        train_arr,test_arr,preprocessor_path = data_tranformation.initiate_data_transformation(train_data_path,test_data_path)
        
        #Model training code
        model_trianer = ModelTrainer()
        print(model_trianer.initiate_model_trainer(train_arr,test_arr))
      
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
