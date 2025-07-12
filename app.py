import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle


# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("💻 Laptop Price Predictor")
st.markdown("Enter laptop specs to get an estimated price.")

# Input Sidebar
company = st.sidebar.selectbox("Company", ['Asus', 'HP', 'Dell', 'Acer', 'Lenovo', 'Apple', 'MSI', 'Toshiba', 'Samsung', 'Microsoft'])
typename = st.sidebar.selectbox("Laptop Type", ['Notebook', 'Gaming', 'Ultrabook', '2 in 1 Convertible', 'Workstation', 'Netbook'])
inches = st.sidebar.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, step=0.1)
screen_resolution = st.sidebar.text_input("Screen Resolution", value="1920x1080")
cpu = st.sidebar.selectbox("CPU", ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other Intel CPU', 'AMD Ryzen 5', 'AMD Ryzen 7'])
gpu = st.sidebar.selectbox("GPU Brand", ['Intel', 'Nvidia', 'AMD'])
opsys = st.sidebar.selectbox("Operating System", ['Windows', 'MacOS', 'Linux/Android/no os/Other'])
weight = st.sidebar.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
ram = st.sidebar.selectbox("RAM (GB)", [4, 8, 12, 16, 32, 64])
touchscreen = st.sidebar.selectbox("Touchscreen", ['No', 'Yes']) == 'Yes'
ips = st.sidebar.selectbox("IPS Display", ['No', 'Yes']) == 'Yes'
memory = st.sidebar.text_input("Memory (e.g., '256GB SSD + 1TB HDD')", value="512GB SSD + 1TB HDD")

# --- Utility Functions ---

def parse_memory(mem_str):
    df = pd.DataFrame([{"Memory": mem_str}])
    df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
    df["Memory"] = df["Memory"].str.replace('GB', '', regex=False)
    df["Memory"] = df["Memory"].str.replace('TB', '000', regex=False)

    new = df["Memory"].str.split("+", n=1, expand=True)
    df["first"] = new[0].str.strip()
    df["second"] = new[1] if new.shape[1] > 1 else pd.Series(["0"] * len(df))
    df["second"] = df["second"].fillna("0").str.strip()

    df["Layer1HDD"] = df["first"].str.contains("HDD", case=False).astype(int)
    df["Layer1SSD"] = df["first"].str.contains("SSD", case=False).astype(int)
    df["Layer2HDD"] = df["second"].str.contains("HDD", case=False).fillna(False).astype(int)
    df["Layer2SSD"] = df["second"].str.contains("SSD", case=False).fillna(False).astype(int)

    df['first'] = df['first'].str.extract(r'(\d+)').fillna('0')
    df['second'] = df['second'].str.extract(r'(\d+)').fillna('0')

    df["first"] = df["first"].astype(int)
    df["second"] = df["second"].astype(int)

    df["HDD"] = df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"]
    df["SSD"] = df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"]

    return df["HDD"].iloc[0], df["SSD"].iloc[0]

def calculate_ppi(screen_res, inches):
    try:
        new = screen_res.split('x', 1)
        x_res = new[0]
        y_res = new[1]
        x_res = re.sub(',', '', x_res)
        x_res = re.findall(r'(\d+\.?\d+)', x_res)[0]
        x_res = int(float(x_res))
        y_res = int(y_res)
        return ((x_res ** 2 + y_res ** 2) ** 0.5) / float(inches)
    except:
        return None

def fetch_cpu(cpu):
    if cpu in ['Intel Core i3', 'Intel Core i5', 'Intel Core i7']:
        return cpu
    elif 'Intel' in cpu:
        return 'Other Intel Cpu'
    else:
        return 'AMD Cpu'

def cat_os(os):
    if os in ['Windows 7', 'Windows 10', 'Windows 10 S']:
        return 'Windows'
    elif os in ['macOS', 'Mac OS X']:
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

# Prediction Button
if st.button("Predict Laptop Price"):

    hdd, ssd = parse_memory(memory)
    ppi = calculate_ppi(screen_resolution, inches)

    if ppi is None:
        st.error("Invalid screen resolution or inches. Please check input.")
    else:
        gpu_brand = gpu.split()[0]
        if gpu_brand == 'ARM':
            gpu_brand = 'Intel'

        df = pd.DataFrame([{
            'Company': company,
            'TypeName': typename,
            'Inches': inches,
            'ScreenResolution': screen_resolution,
            'Cpu brand': fetch_cpu(cpu),
            'Gpu brand': gpu_brand,
            'os': cat_os(opsys),
            'Weight': float(weight),
            'Ram': ram,
            'Touchscreen': bool(touchscreen),
            'IPS display': bool(ips),
            'HDD': int(hdd),
            'SSD': int(ssd),
            'PPI': float(ppi)
        }])

        pred_price = float(np.exp(model.predict(df)[0]))
        st.success(f" Estimated Laptop Price: ₹ {pred_price:,.2f}")
