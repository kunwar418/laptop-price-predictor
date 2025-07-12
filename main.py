from fastapi import FastAPI
import re
from pydantic import BaseModel , Field 
import pickle
from typing import Annotated , Literal , List
import pandas as pd
from fastapi.responses import JSONResponse
import traceback
import numpy as np


# with command would close the file after reading
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# FastAPI object
app = FastAPI()

class UserInput(BaseModel):
    Company: Annotated[str , Field(...,description='brand of laptop',examples=['Asus'])]
    TypeName : Annotated[Literal['Notebook','Gaming','Ultrabook','2 in 1 Convertible','Workstation','Netbook'],Field(...,description='TypeName of laptop')] 
    Inches : Annotated[float,Field(...,gt=0,description='screen size of laptop', examples=[15.6])]
    ScreenResolution :  Annotated[str ,  Field(...,description='Screen Resolution',examples=['Full HD 1920x1080'])]
    Cpu :  Annotated[str , Field(...,description='laptop processor ',examples=['Intel Core i5'])]
    Gpu :  Annotated[Literal['Intel','Nvidia','AMD'],Field(...,description='gpu brand')]
    OpSys : Annotated[Literal['Windows','MacOS','Linux/Android/no os/Other'],Field(...,description='operating system of laptop')]
    Weight:  Annotated[float , Field(...,gt=0,description='Weight of laptop in kg',examples=[1.9])]
    Ram : Annotated[int,Field(...,description='ram of laptop')]
    Touchscreen : Annotated[bool,Field(...,description='laptop has touchscreen or not ')]
    Memory: Annotated[str, Field(..., description='RAM storage like "256GB SSD + 1TB HDD"')]
    Ips: Annotated[bool, Field(..., description='Does screen have IPS?')]


class MemInput(BaseModel):
    Memory: str
   
def calculate_PPI(ScreenResolution : str, Inches : float) -> float :
    try:
        new = ScreenResolution.split('x', 1)
        x_res = new[0]
        y_res = new[1]
        x_res = re.sub(',', '', x_res)
        x_res = re.findall(r'(\d+\.?\d+)', x_res)[0]
        x_res = int(float(x_res))
        y_res = int(y_res)
        PPI = ((x_res**2 + y_res**2)**0.5) / float(Inches)
        return PPI
    except:
        return None
            
def fetch_Cpu(Cpu :  str) -> str:
    if Cpu in ['Intel Core i3', 'Intel Core i5', 'Intel Core i7']:
        return Cpu
   
    elif 'Intel' in Cpu:
        return 'Other Intel Cpu'
    else:
        return 'AMD Cpu'


    
def cat_os(OpSys):
    if OpSys in ['Windows 7','Windows 10','Windows 10 S']:
        return 'Windows'
    elif OpSys in ['macOS','Mac OS X']:
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

   
def parse_memory(mem_list: List[MemInput]):
    df = pd.DataFrame([m.model_dump() for m in mem_list])

    # Step 1: Clean 'Memory' column
    df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
    df["Memory"] = df["Memory"].str.replace('GB', '', regex=False)
    df["Memory"] = df["Memory"].str.replace('TB', '000', regex=False)

    # Step 2: Split into two parts
    new = df["Memory"].str.split("+", n=1, expand=True)
    df["first"] = new[0].str.strip()
    df["second"] = new[1] if new.shape[1] > 1 else pd.Series(["0"] * len(df))
    df["second"] = df["second"].fillna("0").str.strip()


    # Step 3: Identify storage TypeName in both parts
    df["Layer1HDD"] = df["first"].str.contains("HDD", case=False).astype(int)
    df["Layer1SSD"] = df["first"].str.contains("SSD", case=False).astype(int)

    df["Layer2HDD"] = df["second"].str.contains("HDD", case=False).fillna(False).astype(int)
    df["Layer2SSD"] = df["second"].str.contains("SSD", case=False).fillna(False).astype(int)

    # Step 4: Remove all non-digits to extract only numbers
    df['first'] = df['first'].str.extract(r'(\d+)').fillna('0')
    df['second'] = df['second'].str.extract(r'(\d+)').fillna('0')

    # Step 5: Convert to int
    df["first"] = df["first"].astype(int)
    df["second"] = df["second"].astype(int)

    # Step 6: Final memory breakdown
    df["HDD"] = df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"]
    df["SSD"] = df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"]

    # Step 7: Drop unused
    df.drop(columns=[
        'first', 'second',
        'Layer1HDD', 'Layer1SSD',
        'Layer2HDD', 'Layer2SSD'
    ], inplace=True)

    return df[["HDD", "SSD"]].to_dict(orient="records")


@app.get("/")
def root():
    return {"message": "Laptop Price Prediction API is live "}


@app.post('/predict')
def price_predict(data : UserInput):
    try :
        mem_out = parse_memory([MemInput(Memory=data.Memory)])[0]
        ppi_val = calculate_PPI(data.ScreenResolution, data.Inches)
        if ppi_val is None:
            return JSONResponse(status_code=400, content={"error": "Invalid ScreenResolution or Inches for PPI"})

        gpu_brand = data.Gpu.split()[0]
        if gpu_brand == 'ARM':
            gpu_brand = 'Intel'
        input_df = pd.DataFrame([{
            'Company' : data.Company,
            'TypeName'     : data.TypeName,
            'Inches'  :data.Inches,
            'ScreenResolution'  : data.ScreenResolution,
            'Cpu brand'  : fetch_Cpu(data.Cpu),
            'Gpu brand' : gpu_brand, 
            'os'    : cat_os(data.OpSys), 
            'Weight' : float(data.Weight),
            'Ram': data.Ram,   
            'Touchscreen': bool(data.Touchscreen),    
            'IPS display': bool(data.Ips),    
            'HDD': int(mem_out['HDD']),   
            'SSD': int(mem_out['SSD']),
            'PPI' : ppi_val   
        }])  
        

        prediction = float(np.exp(model.predict(input_df)[0]))

        return JSONResponse(status_code=200, content={'predicted_price': prediction})

    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e), 'trace': traceback.format_exc()})