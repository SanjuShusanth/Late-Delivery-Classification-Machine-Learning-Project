import os
from MLProject import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from MLProject.entity.config_entity import DataTransformationConfig
from MLProject.utils.common import save_object
import pandas as pd
import scipy
import numpy as np
import pickle


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformation_object(self):
        try:
            Numerical_cols = ['Days_for_shipment_(scheduled)', 'Benefit_per_order',
                              'Sales_per_customer', 'Latitude',
                              'Longitude', 'Order_Item_Discount', 'Order_Item_Discount_Rate',
                              'Order_Item_Product_Price', 'Order_Item_Profit_Ratio',
                              'Order_Item_Quantity', 'Sales', 'Order_Item_Total',
                              'Order_Profit_Per_Order', 'Product_Price']
            
            nom_cat_cols = ['Type','Order_Status','Shipping_Mode']
            
            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median'))
                ]
            )

            # Nominal_Categorigal Pipeline
            nom_cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder', OneHotEncoder(drop='first'))
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,Numerical_cols),
                ('nom_cat_pipeline',nom_cat_pipeline,nom_cat_cols)
            ])

            return preprocessor

        except Exception as e:
            raise e
        
    def initiate_data_transformation(self):
        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)

        num_cols_train = train_df.select_dtypes(np.number).columns
        num_cols_test = test_df.select_dtypes(np.number).columns

        for col in num_cols_train:
            Q1 = train_df[col].quantile(0.25)
            Q3 = train_df[col].quantile(0.75)
            IQR = Q3 - Q1
        
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        
            train_df = train_df[(train_df[col] >= lower_bound) & (train_df[col] <= upper_bound)]

        for col in num_cols_test:
            Q1 = test_df[col].quantile(0.25)
            Q3 = test_df[col].quantile(0.75)
            IQR = Q3 - Q1
        
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        
            test_df = test_df[(test_df[col] >= lower_bound) & (test_df[col] <= upper_bound)]

        columns_to_drop = ['Order_City', 'Order_Country', 'Order_State', 'Order_Region', 'Delivery_Status', 'Market' ]

        train_df = train_df.drop(columns=columns_to_drop)
        test_df = test_df.drop(columns=columns_to_drop)


        train_df['Shipping_Mode']= train_df['Shipping_Mode'].replace('Same Day', 'Premium Class')
        train_df['Order_Item_Discount_Rate']= train_df['Order_Item_Discount_Rate'].apply(lambda x: x*100)

        test_df['Shipping_Mode']= test_df['Shipping_Mode'].replace('Same Day', 'Premium Class')
        test_df['Order_Item_Discount_Rate']= test_df['Order_Item_Discount_Rate'].apply(lambda x: x*100)


        input_feature_train_df = train_df.drop(columns=['Late_delivery_risk'])
        target_feature_train_df = train_df['Late_delivery_risk']

        input_feature_test_df = test_df.drop(columns=['Late_delivery_risk'])
        target_feature_test_df = test_df['Late_delivery_risk']

        preprocessing_obj = self.get_data_transformation_object()

        input_feature_train_arr = pd.DataFrame(preprocessing_obj.fit_transform(input_feature_train_df), columns=preprocessing_obj.get_feature_names_out())
        input_feature_test_arr = pd.DataFrame(preprocessing_obj.fit_transform(input_feature_test_df), columns=preprocessing_obj.get_feature_names_out())

        train_arr = pd.concat([input_feature_train_arr, target_feature_train_df], axis=1)
        test_arr = pd.concat([input_feature_test_arr, target_feature_test_df], axis=1)
        
        train_arr.columns = ['Days_for_shipment_(scheduled)','Benefit_per_order','Sales_per_customer','Latitude','Longitude',
                             'Order_Item_Discount','Order_Item_Discount_Rate','Order_Item_Product_Price',
                             'Order_Item_Profit_Ratio','Order_Item_Quantity','Sales','Order_Item_Total',
                             'Order_Profit_Per_Order','Product_Price','Type_DEBIT','Type_PAYMENT',
                             'Type_TRANSFER','Order_Status_CLOSED','Order_Status_COMPLETE','Order_Status_ON_HOLD',
                             'Order_Status_PAYMENT_REVIEW','Order_Status_PENDING','Order_Status_PENDING_PAYMENT',
                             'Order_Status_PROCESSING','Order_Status_SUSPECTED_FRAUD','Shipping_Mode_Premium_Class',
                             'Shipping_Mode_Second_Class','Shipping_Mode_Standard_Class','Late_delivery_risk']
        
        test_arr.columns = ['Days_for_shipment_(scheduled)','Benefit_per_order','Sales_per_customer','Latitude','Longitude',
                             'Order_Item_Discount','Order_Item_Discount_Rate','Order_Item_Product_Price',
                             'Order_Item_Profit_Ratio','Order_Item_Quantity','Sales','Order_Item_Total',
                             'Order_Profit_Per_Order','Product_Price','Type_DEBIT','Type_PAYMENT',
                             'Type_TRANSFER','Order_Status_CLOSED','Order_Status_COMPLETE','Order_Status_ON_HOLD',
                             'Order_Status_PAYMENT_REVIEW','Order_Status_PENDING','Order_Status_PENDING_PAYMENT',
                             'Order_Status_PROCESSING','Order_Status_SUSPECTED_FRAUD','Shipping_Mode_Premium_Class',
                             'Shipping_Mode_Second_Class','Shipping_Mode_Standard_Class','Late_delivery_risk']

        train_arr.to_csv(os.path.join(self.config.root_dir, 'trans_train.csv'), index=False)
        test_arr.to_csv(os.path.join(self.config.root_dir, 'trans_test.csv'), index=False)

        save_object(
                file_path=self.config.preprocessor_path,
                obj=preprocessing_obj
            )

        return (
                train_arr,
                test_arr,
                self.config.preprocessor_path
            )