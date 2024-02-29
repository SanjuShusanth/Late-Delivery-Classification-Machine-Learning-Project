import os
from MLProject import logger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from MLProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self)->bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)

            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()


            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f'Validation status: {validation_status}')

                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f'Validation status: {validation_status}')

            return validation_status
        
        except Exception as e:
            raise e
        
    
    def initiate_data_split(self):
        try:
            df = pd.read_csv(self.config.unzip_data_dir)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            return(
                self.config.train_data_path, 
                self.config.test_data_path
            )
        
        except Exception as e:
            raise e