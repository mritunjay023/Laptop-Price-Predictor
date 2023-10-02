import numpy as np
import pandas as pd
import streamlit as st

# Load the pre-trained model and data
pipe = pd.read_pickle(open('pipe.pkl', 'rb'))
df = pd.read_pickle(open('df.pkl', 'rb'))

# Create a dictionary to store user inputs
user_inputs = {}

# Set page title
st.title("Laptop Price Predictor")

# Sidebar with user inputs
st.sidebar.header("User Inputs")

# Brand
company = st.sidebar.selectbox('Brand', df['Company'].unique())
user_inputs['Brand'] = company

# Type of laptop
type = st.sidebar.selectbox('Type', df['TypeName'].unique())
user_inputs['Type'] = type

# RAM
ram = st.sidebar.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
user_inputs['RAM'] = ram

# Weight
weight = st.sidebar.number_input('Weight of the Laptop (kg)', min_value=0.1)
user_inputs['Weight'] = weight

# Touchscreen
touchscreen = st.sidebar.radio('Touchscreen', ['No', 'Yes'])
user_inputs['Touchscreen'] = touchscreen

# IPS
ips = st.sidebar.radio('IPS', ['No', 'Yes'])
user_inputs['IPS'] = ips

# Screen size
screen_size = st.sidebar.number_input('Screen Size (in inches)', min_value=0.1)
user_inputs['Screen Size'] = screen_size

# Resolution
resolution = st.sidebar.selectbox('Screen Resolution',
                                  ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800',
                                   '2560x1600', '2560x1440', '2304x1440'])
user_inputs['Resolution'] = resolution

# CPU
cpu = st.sidebar.selectbox('CPU', df['Cpu brand'].unique())
user_inputs['CPU'] = cpu

# HDD
hdd = st.sidebar.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
user_inputs['HDD'] = hdd

# SSD
ssd = st.sidebar.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
user_inputs['SSD'] = ssd
if hdd == 0 and ssd == 0:
    st.sidebar.error("Both SSD and HDD cannot be zero. Please select at least one storage option.")
# GPU
gpu = st.sidebar.selectbox('GPU', df['Gpu brand'].unique())
user_inputs['GPU'] = gpu

# OS
os = st.sidebar.selectbox('OS', df['os'].unique())
user_inputs['OS'] = os

# Full HD
Full_HD = st.sidebar.radio('FULL HD', ['No', 'Yes'])
user_inputs['Full HD'] = Full_HD

if st.sidebar.button('Predict Price'):
    # Query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    if Full_HD == 'Yes':
        Full_HD = 1
    else:
        Full_HD = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    ram = np.log(ram)
    weight = np.log(weight)
    query = np.array([company, type, ram, weight, touchscreen, ips, Full_HD, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 13)
    predicted_price = np.exp(pipe.predict(query)[0])

    # Display the predicted price
    st.header("Predicted Price")

    st.markdown(f'<p style="font-size:24px;font-weight:bold;">Rs. {predicted_price:.2f}</p>', unsafe_allow_html=True)


# Reset button at the end of the main content
if st.button('Reset'):
    user_inputs.clear()
