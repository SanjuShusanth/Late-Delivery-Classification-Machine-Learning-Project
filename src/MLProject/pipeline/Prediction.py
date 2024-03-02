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
            
            data_values = data.iloc[0:16].to_numpy()  # Assuming the first 17 elements correspond to your features
            
            data_df = pd.DataFrame(data_values, columns=['Type','Days_for_shipment_scheduled','Late_delivery_risk','Customer_Country',
                                                         'Customer_Segment','Order_Country','Order_Item_Discount_Rate','Order_Item_Quantity',
                                                         'Sales','Order_Status','Product_Name','Product_Price','Shipping_Mode','order_date_year',
                                                         'order_date_month','order_date_day'])


            #data_values = data_df.iloc[0].to_numpy()

            data_scaled = preprocessor.transform(data_df)
            pred = model.predict(data_scaled)

            if (pred[0] == 0):
                print("No Late delivery risk")
            else:
                print("Late Delivery risk")


        except Exception as e:
            logger.info('Exception occurred in prediction')
            raise e