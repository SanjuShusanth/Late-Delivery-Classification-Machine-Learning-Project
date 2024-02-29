import pandas as pd
import os
from MLProject import logger
import xgboost
from xgboost import XGBClassifier
import pickle
from MLProject.utils.common import save_object
import joblib
from MLProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_data = train_data.dropna(subset=[self.config.target_column])
        test_data = test_data.dropna(subset=[self.config.target_column])


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        xgb = XGBClassifier(max_depth = self.config.max_depth, learning_rate = self.config.learning_rate, 
                           n_estimators=self.config.n_estimators,random_state=42)
        
        xgb.fit(train_x, train_y)

        joblib.dump(xgb, os.path.join(self.config.root_dir, self.config.model_name))

        save_object(
            file_path = self.config.model_path,
            obj = xgb
        )

        return self.config.model_path