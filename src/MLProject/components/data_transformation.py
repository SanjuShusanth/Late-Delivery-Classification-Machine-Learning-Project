import os
from MLProject import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from MLProject.entity.config_entity import DataTransformationConfig
from MLProject.utils.common import save_object
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import scipy
import numpy as np
import pickle


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformation_object(self):
        try:
            Numerical_cols = ['Days_for_shipment_scheduled',
                              'Order_Item_Discount_Rate', 'Order_Item_Quantity', 'Sales',
                              'Product_Price', 'order_date_year', 'order_date_month','order_date_day']
            
            nom_cat_cols = ['Customer_Country','Order_Country']
            
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

        train_df['Order_Item_Discount_Rate'] = train_df['Order_Item_Discount_Rate'].apply(lambda x: x*100)
        test_df['Order_Item_Discount_Rate'] = test_df['Order_Item_Discount_Rate'].apply(lambda x: x*100)

        # Order Country train
        dict_country_train = train_df['Order_Country'].value_counts().to_dict()
        
        new_country_train = []
        for k, v in dict_country_train.items():
            if v < 4000:
                new_country_train.append(k)
        
        lst_index_country_train = train_df[train_df['Order_Country'].isin(new_country_train)].index.to_list()
        train_df.loc[lst_index_country_train, 'Order_Country'] = 'Others'


        # Order Country test 
        dict_country_test = test_df['Order_Country'].value_counts().to_dict()

        new_country_test = []
        for k, v in dict_country_test.items():
                if v < 1500:
                    new_country_test.append(k)

        lst_index_country_test = test_df[test_df['Order_Country'].isin(new_country_test)].index.to_list()
        test_df.loc[lst_index_country_test, 'Order_Country'] = 'Others'

        # Product name train

        dict_product_train = train_df['Product_Name'].value_counts().to_dict()

        new_product_train = []
        for k, v in dict_product_train.items():
            if v < 10000:
                new_product_train.append(k)

        lst_index_product_train = train_df[train_df['Product_Name'].isin(new_product_train)].index.to_list()
        train_df.loc[lst_index_product_train, 'Product_Name'] = 'Others'

        # Product Name test

        dict_product_test = test_df['Product_Name'].value_counts().to_dict()

        new_product_test = []
        for k, v in dict_product_test.items():
            if v < 4500:
                new_product_test.append(k)

        lst_index_product_test = test_df[test_df['Product_Name'].isin(new_product_test)].index.to_list()
        test_df.loc[lst_index_product_test, 'Product_Name'] = 'Others'

        train_df['Product_Name'] = train_df['Product_Name'].replace("Nike Men's CJ Elite 2 TD Football Cleat", 'Nike Mens CJ Elite 2 TD Football Cleat')
        test_df['Product_Name'] = test_df['Product_Name'].replace("Nike Men's CJ Elite 2 TD Football Cleat", 'Nike Mens CJ Elite 2 TD Football Cleat')

        train_df['Product_Name'] = train_df['Product_Name'].replace("Nike Men's Dri-FIT Victory Golf Polo",'Nike Mens Dri-FIT Victory Golf Polo')
        test_df['Product_Name'] = test_df['Product_Name'].replace("Nike Men's Dri-FIT Victory Golf Polo",'Nike Mens Dri-FIT Victory Golf Polo')
        
        train_df['Product_Name'] = train_df['Product_Name'].replace("O'Brien Men's Neoprene Life Vest", 'O Brien Mens Neoprene Life Vest')
        test_df['Product_Name'] = test_df['Product_Name'].replace("O'Brien Men's Neoprene Life Vest", 'O Brien Mens Neoprene Life Vest')

        train_df['Product_Name'] = train_df['Product_Name'].replace("Diamondback Women's Serene Classic Comfort Bi", 'Diamondback Womens Serene Classic Comfort Bi')
        test_df['Product_Name'] = test_df['Product_Name'].replace("Diamondback Women's Serene Classic Comfort Bi", 'Diamondback Womens Serene Classic Comfort Bi')

        train_df['Product_Name'] = train_df['Product_Name'].replace("Nike Men's Free 5.0+ Running Shoe",'Nike Mens Free 5.0+ Running Shoe')
        test_df['Product_Name'] = test_df['Product_Name'].replace("Nike Men's Free 5.0+ Running Shoe",'Nike Mens Free 5.0+ Running Shoe')

        train_df['Product_Name'] = train_df['Product_Name'].replace("Under Armour Girls' Toddler Spine Surge Runni",'Under Armour Girls Toddler Spine Surge Runni')
        test_df['Product_Name'] = test_df['Product_Name'].replace("Under Armour Girls' Toddler Spine Surge Runni",'Under Armour Girls Toddler Spine Surge Runni')

        Numerical_cols_train = ['Days_for_shipment_scheduled','Order_Item_Discount_Rate', 'Order_Item_Quantity', 'Sales',
                                'Product_Price', 'order_date_year', 'order_date_month','order_date_day']

        Numerical_cols_test = ['Days_for_shipment_scheduled','Order_Item_Discount_Rate', 'Order_Item_Quantity', 'Sales',
                               'Product_Price', 'order_date_year', 'order_date_month','order_date_day']
        

        for column in Numerical_cols_train:
            Q1 = train_df[column].quantile(0.25)
            Q3 = train_df[column].quantile(0.75)
            IQR = Q3 - Q1
        
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        
            train_df = train_df[(train_df[column] >= lower_bound) & (train_df[column] <= upper_bound)]


        for column in Numerical_cols_test:
            Q1 = test_df[column].quantile(0.25)
            Q3 = test_df[column].quantile(0.75)
            IQR = Q3 - Q1
        
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        
            test_df = test_df[(test_df[column] >= lower_bound) & (test_df[column] <= upper_bound)]

        
        lab_cat_cols_train = ['Order_Status','Product_Name']
        lab_cat_cols_test = ['Order_Status','Product_Name']

        nom_cat_cols_train = ['Customer_Country','Order_Country']
        nom_cat_cols_test = ['Customer_Country','Order_Country']

        ord_cat_cols_train = ['Type', 'Customer_Segment', 'Shipping_Mode']
        ord_cat_cols_test = ['Type', 'Customer_Segment', 'Shipping_Mode']

        Type = {'DEBIT':1,'TRANSFER':2,'PAYMENT':3, 'CASH':4}
        customer = {'Consumer':1,'Corporate':2,'Home Office':3}
        Shipping = {'Standard Class':1, 'Second Class':2,'First Class':3,'Same Day':4}

        train_df['Type'] = train_df['Type'].map(Type)
        test_df['Type'] = test_df['Type'].map(Type)

        train_df['Customer_Segment'] = train_df['Customer_Segment'].map(customer)
        test_df['Customer_Segment'] = test_df['Customer_Segment'].map(customer)

        train_df['Shipping_Mode'] = train_df['Shipping_Mode'].map(Shipping)
        test_df['Shipping_Mode'] = test_df['Shipping_Mode'].map(Shipping)


        encoding_cols =['Order_Status','Product_Name']
        label_encoders ={}
        for column in encoding_cols:
            label_encoders[column]=LabelEncoder()
            train_df[column]=label_encoders[column].fit_transform(train_df[column])
            test_df[column]=label_encoders[column].transform(test_df[column])

        preprocessing_obj = self.get_data_transformation_object()

        input_feature_train_df = train_df.drop(columns=['Late_delivery_risk'])
        target_feature_train_df = train_df['Late_delivery_risk']

        input_feature_test_df = test_df.drop(columns=['Late_delivery_risk'])
        target_feature_test_df = test_df['Late_delivery_risk']

        input_feature_train_arr = pd.DataFrame(preprocessing_obj.fit_transform(input_feature_train_df), columns=preprocessing_obj.get_feature_names_out())
        input_feature_test_arr = pd.DataFrame(preprocessing_obj.transform(input_feature_test_df), columns=preprocessing_obj.get_feature_names_out())

        train_arr = pd.concat([input_feature_train_arr, target_feature_train_df], axis=1)
        test_arr = pd.concat([input_feature_test_arr, target_feature_test_df], axis=1)
        
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