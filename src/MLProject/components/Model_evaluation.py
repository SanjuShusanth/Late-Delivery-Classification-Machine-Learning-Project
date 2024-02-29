import os
import pandas as pd
from sklearn.metrics import  accuracy_score, precision_score, recall_score, roc_auc_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from MLProject.utils.common import save_json
from MLProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)

        return accuracy, precision, recall
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        test_data = test_data.dropna(subset=[self.config.target_column])
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_quantities = model.predict(test_x)
            (accuracy, precision, recall) = self.eval_metrics(test_y, predicted_quantities)

            # saving metric as local

            scores = {'accuracy': accuracy, 'precision': precision, 'recall':recall}
            save_json(path= Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('recall', recall)

            if tracking_url_type_store != 'file':


                mlflow.sklearn.log_model(model, 'model', registered_model_name='XgboostClassificationModel')

            else:
                mlflow.sklearn.log_model(model, 'model')