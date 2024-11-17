
import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
from catboost import CatBoostRegressor
from sklearn. ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    )
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from src.mlproject.exception import CustomException
from src.mlproject.logger import log as logging

from src.mlproject.utils import save_object,evaluate_models
import mlflow

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def eval_metrics(self, y_true, y_pred):
        try:
            # RMSE calculation
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # MAE calculation
            mae = mean_absolute_error(y_true, y_pred)
            
            # R2 calculation
            r2 = r2_score(y_true, y_pred)

            return rmse, mae, r2
        
        except Exception as e:
            raise CustomException(f"Error in evaluation metrics: {str(e)}", sys)
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            logging.info("Splitting training and test data")
            
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                        
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report = evaluate_models(X_train,y_train,X_test,y_test,models,params)
            
            #To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            #To get best model name
            best_model_name = list (model_report.keys())[
                list (model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print("This is the best model")
            print(best_model_name)
            
            model_names = list(params.keys())
            
            actuall_model=""
            
            for model in model_names:
                if best_model_name == model:
                    actuall_model = actuall_model + model
                    
            best_params = params[actuall_model]
            
            mlflow.set_registry_uri("https://dagshub.com/kaustubhkaushik26/MLproject.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            # ML FLOW 
            input_example = X_train[0].reshape(1, -1)  # Select a single sample and reshape if needed
            
            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Log the model with input example
                mlflow.sklearn.log_model(best_model, "model", input_example=input_example)

                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")
            
            
            
            if best_model_score<0.6:
                raise CustomException("No best Model found")
            logging.info(f"Best found model on training and test dataset ")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score (y_test, predicted)
            return r2_square
        
        
        
        except Exception as e:
            raise CustomException(e,sys)
    