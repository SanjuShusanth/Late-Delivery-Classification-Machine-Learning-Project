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
        Late_delivery_risk: str = Form(...),
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

    # Mapping logic for categorical variables
    Type_mapped = Type_mapping.get(Type, Type)
    Shipping_Mode_mapped = Shipping_Mode_mapping.get(Shipping_Mode, Shipping_Mode)
    Order_Status_mapped = Order_status_mapping.get(Order_Status, Order_Status)

    data_df = pd.DataFrame({
        'Type': [Type_mapped],
        'Days_for_shipment_scheduled': [Days_for_shipment_scheduled],
        'Late_delivery_risk': [Late_delivery_risk],
        'Customer_Country': [Customer_Country],
        'Customer_Segment': [Customer_Segment],
        'Order_Country': [Order_Country],
        'Order_Item_Discount_Rate': [Order_Item_Discount_Rate],
        'Order_Item_Quantity': [Order_Item_Quantity],
        'Sales': [Sales],
        'Order_Status': [Order_Status_mapped],
        'Product_Name': [Product_Name],
        'Product_Price': [Product_Price],
        'Shipping_Mode': [Shipping_Mode_mapped],
        'order_date_year': [order_date_year],
        'order_date_month': [order_date_month],
        'order_date_day': [order_date_day]
    })

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8080)