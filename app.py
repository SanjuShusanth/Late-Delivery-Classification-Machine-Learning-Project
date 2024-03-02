import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from MLProject.pipeline.Prediction import PredictionPipeline
import pandas as pd
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi import Depends
from MLProject import logger

app = FastAPI()

app.mount("/static", StaticFiles(directory="templates/static"), name="static")

templates = Jinja2Templates(directory='templates')


@app.get('/train')
async def training():
    os.system('python main.py')
    return "Training Successful"

@app.get("/home/", response_class=HTMLResponse)
async def hello(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/predict/', response_class=HTMLResponse)
async def prediction(request: Request,
        Type: str = Form(...),
        Days_for_shipment_scheduled: str = Form(...),
        Customer_Country: str = Form(...),
        Customer_Segment: str = Form(...),
        Order_Country: str = Form(...),
        Order_Item_Discount_Rate: str = Form(...),
        Order_Item_Quantity: str = Form(...),
        Sales: str = Form(...),
        Order_Status: str = Form(...),
        Product_Name: str = Form(...),
        Product_Price: str = Form(...),
        Shipping_Mode: str = Form(...),
        order_date_year: str = Form(...),
        order_date_month: str = Form(...),
        order_date_day: str = Form(...)
):      
        Type_mapping = {
             "DEBIT":"DEBIT",
             "TRANSFER":"TRANSFER",
             "PAYMENT":"PAYMENT",
             "CASH":"CASH"
        }
        
        Shipping_Mode_maping = {
             "Same Day": "Same Day",
             "First Class": "First Class",
             "Standard Class":"Standard Class",
             "Second Class":"Second Class"
        }

        Order_Status_maping = {
             "COMPLETE":"COMPLETE",
             "PROCESSING":"PROCESSING",
             "CLOSED":"CLOSED",
             "ON_HOLD":"ON_HOLD",
             "PENDING":"PENDING",
             "PAYMENT_REVIEW":"PAYMENT_REVIEW",
             "CANCELED":"CANCELED",
             "SUSPECTED_FRAUD":"SUSPECTED_FRAUD",
             "PENDING_PAYMENT":"PENDING_PAYMENT"
        }

        Customer_Country_mapping = {
             "EE. UU.":"EE. UU.",
             "Puerto Rico":"Puerto Rico"
        }

        Customer_Segment_mapping = {
             "Consumer":"Consumer",
             "Corporate":"Corporate",
             "Home Office":"Home Office"
        }

        Order_Country_mapping = {
             "Estados Unidos":"Estados Unidos",
             "México":"México",
             "Francia":"Francia",
             "Alemania":"Alemania",
             "Australia":"Australia",
             "Brasil":"Brasil",
             "Reino Unido":"Reino Unido",
             "China":"China",
             "Others":"Others"
        }

        Product_Name_mapping = {
             "Perfect Fitness Perfect Rip Deck":"Perfect Fitness Perfect Rip Deck",
             "Nike Mens Dri-FIT Victory Golf Polo":"Nike Mens Dri-FIT Victory Golf Polo",
             "Nike Mens CJ Elite 2 TD Football Cleat":"Nike Mens CJ Elite 2 TD Football Cleat",
             "O'Brien Men's Neoprene Life Vest":"O Brien Mens Neoprene Life Vest",
             "Field & Stream Sportsman 16 Gun Fire Safe":"Field & Stream Sportsman 16 Gun Fire Safe",
             "Pelican Sunstream 100 Kayak":"Pelican Sunstream 100 Kayak",
             "Others":"Others"
        }

    
        Type_mapped = Type_mapping.get(Type, Type)
        Shipping_mapped = Shipping_Mode_maping.get(Shipping_Mode, Shipping_Mode)
        Order_Status_mapped = Order_Status_maping.get(Order_Status, Order_Status)
        Customer_Country_mapped = Customer_Country_mapping.get(Customer_Country, Customer_Country)
        Customer_Segment_mapped = Customer_Segment_mapping.get(Customer_Segment, Customer_Segment)
        Order_Country_mapped = Order_Country_mapping.get(Order_Country, Order_Country)
        Product_Name_mapped = Product_Name_mapping.get(Product_Name, Product_Name)


        test_data = [Type_mapped, Shipping_mapped, Order_Status_mapped, Customer_Country_mapped,Customer_Segment_mapped,
                     Order_Country_mapped, Product_Name_mapped,Days_for_shipment_scheduled,Order_Item_Discount_Rate,
                     Order_Item_Quantity,Sales,Product_Price,order_date_year, order_date_month, order_date_day]
        
        columns = ['Type','Days_for_shipment_scheduled','Customer_Country',
                   'Customer_Segment','Order_Country','Order_Item_Discount_Rate','Order_Item_Quantity',
                   'Sales','Order_Status','Product_Name','Product_Price','Shipping_Mode','order_date_year',
                   'order_date_month','order_date_day']
        
        input_df = pd.DataFrame([test_data], columns=columns)

        pipeline = PredictionPipeline()

        pred = pipeline.predict(input_df)

        if (pred[0] == 0):
             Results = "No Late Delivery Risk"
        else:
             Results = "Late Delivery Risk"

        return templates.TemplateResponse('results.html', {'request': request, 'result': Results})

if __name__ == "__main__":
     uvicorn.run(app, host='0.0.0.0', port=8080)