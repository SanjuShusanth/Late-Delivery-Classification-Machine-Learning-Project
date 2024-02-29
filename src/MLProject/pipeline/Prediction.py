import numpy as np
import pickle
import pandas as pd
import os
from pathlib import Path
import pickle
from src.MLProject.utils.common import load_object
from src.MLProject import logger

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, data):
        try:
            preprocess_obj = Path(r'C:\Users\Sanju\WORKSPACE\Late-Delivery-Classification-Machine-Learning-Project\artifacts\models\preprocessor.pkl')
            model_path = Path(r'C:\Users\Sanju\WORKSPACE\Late-Delivery-Classification-Machine-Learning-Project\artifacts\models\model.pkl')
            
            preprocessor = load_object(preprocess_obj)
            model = load_object(model_path)
            
            data_values = data.iloc[0:11].to_numpy()  # Assuming the first 11 elements correspond to your features
            
            data_df = pd.DataFrame(data_values, columns=['Drop_point', 'Shipment_Mode', 'Dosage_Form', 'Line_Item_Quantity',
                                                           'Pack_Price', 'Unit_Price', 'Weight', 'Freight_Cost',
                                                           'Line_Item_Insurance', 'Delivery_Status', 'Pickup_Point'])


            #data_values = data_df.iloc[0].to_numpy()

            data_scaled = preprocessor.transform(data_df)
            pred = model.predict(data_scaled)


            return pred

        except Exception as e:
            logger.info('Exception occurred in prediction')
            raise e