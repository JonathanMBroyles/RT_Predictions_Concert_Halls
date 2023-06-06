#!/usr/bin/env python
# coding: utf-8

# In[1]:


# By Jonathan Broyles (Spring 2023)

import numpy as np
import pandas as pd
import csv
from sklearn.metrics import mean_squared_error


# # Read in the Database

# In[2]:


# Make a DataFrame with the CSV data and obtain basic statistical information

df = pd.read_csv("Final_Halls_Dataset.csv")  # Creates a Pandas DataFrame
print(df.head())  # Shows the first five rows for each column in the DataFrame
# Shows basic statistical results for each column in the DataFrame
print(df.describe())
print(df.shape)  # Shows how many rows and columns are in the DataFrame


# # Classification by Hall Type

# In[3]:


df["Hall Type Number"]


# In[4]:


Hall_labels = ["Shoebox", "Fan", "Oval", "Vineyard", "Hexagon", "Octagon", "Other"]
Hall_numbers = [21,14,6,3,2,2,2]


# # Random Forest Model

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier


df_clean_RF_x = df[["Seating Capacity", "Room Volume m^3", "Surface Area m^2", "Noise Level Number", "Ceiling Alpha", "Wall Alpha","Floor Alpha", "Hall Type Number"]] #, "Stage Area m^2", "Noise Level Number"
df_clean_RF_y = df["RT - 500 Hz Unocc"]


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(df_clean_RF_x, df_clean_RF_y, test_size = 0.3, random_state=1)


# In[9]:


rf_RT = RandomForestRegressor()



rf_RT_best = RandomForestRegressor(bootstrap = True, max_depth = 100, max_features = 3, min_samples_leaf = 5, min_samples_split = 12,
                                   n_estimators = 500)
rf_RT_best.fit(X_train.values, y_train)


y_pred = rf_RT_best.predict(X_test.values)





# # Deployment of the RF Model as an Interactive Design Tool

# In[111]:


from dash import Dash, dcc, html, Input, Output
import plotly.express as px


# In[113]:


import dash_bootstrap_components as dbc


# In[167]:


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(external_stylesheets=[dbc.themes.YETI]) # , external_stylesheets=external_stylesheets
server = app.server

app.layout = html.Div([
    html.H1("Estimating the Mid-Frequency Reverberation Time of Concert Halls \n(Version 1.0)", style = {"textAlign": "center"}),
    
    html.H4("Concert Hall Volume", style = {"textAlign": "center"}),    
    dcc.Slider(2000, 30000, step = 1000,
               marks={
                2000: {'label': '2,000 m^3'},
                5000: '5,000 m^3',
                10000: '10,000 m^3',
                15000: '15,000 m^3',
                20000: '20,000 m^3',
                25000: '25,000 m^3',
                30000: '30,000 m^3'
                },
               value=10000,
               tooltip={"placement": "bottom", "always_visible": True},
               id='Room_Volume'
    ),
    
    html.H4("Concert Hall Surface Area (m^2)", style = {"textAlign": "center"}),    
    dcc.Slider(1000, 10000, 500,
               marks={
                1000: {'label': '1,000 m^2'},
                2500: '2,500 m^2',
                5000: '10,000 m^2',
                7500: '15,000 m^2',
                10000: '20,000 m^2'
                },
               value=2500,
               tooltip={"placement": "bottom", "always_visible": True},
               id='Surface_Area'
    ),
    
    html.H4("Seating Capacity", style = {"textAlign": "center"}),    
    dcc.Slider(250, 2500, 50,
               marks={
                250: {'label': '250'},
                500: '500',
                1000: '1,000',
                1500: '1,500',
                2000: '2,000',
                2500: '2,500'
                },
               value=500,
               tooltip={"placement": "bottom", "always_visible": True},
               id='Seating_Capacity'
    ),
    
    html.H4("Background Noise (NC-#)", style = {"textAlign": "center"}),    
    dcc.Slider(15, 25, 5,
               value=15,
               tooltip={"placement": "bottom", "always_visible": True},
               id='Noise_Level'
    ),
    
    html.H4("Ceiling Finish", style = {"textAlign": "center"}),
    dcc.Dropdown(options=[
       {'label': 'Gypsum Boards', 'value': 0.10},
       {'label': 'Glass Fiber Reinforced Gypsum', 'value': 0.04},
       {'label': 'GRC Panels', 'value': 0.03},
       {'label': 'Concrete Slab', 'value': 0.01},
       {'label': 'Steel Panel', 'value': 0.15},
       ], value=0.1, id='Ceiling_Material'
    ),
    
    html.H4("Wall Finish", style = {"textAlign": "center"}),
    dcc.Dropdown(options=[
       {'label': 'Wood', 'value': 0.10},
       {'label': 'Plywood', 'value': 0.10},
       {'label': 'Gypsum Boards', 'value': 0.10},
       {'label': 'Wood on Concrete', 'value': 0.07},
       {'label': 'Gypsum on Concrete', 'value': 0.05},
       {'label': 'Concrete', 'value': 0.03},
       {'label': 'Tile on Concrete', 'value': 0.03},
       {'label': 'Marble on Concrete', 'value': 0.03},
       {'label': 'Granite Stone Panels', 'value': 0.01},
       {'label': 'Glass Fiber Reinforced Gypsum', 'value': 0.04},
       {'label': 'Glass Fiber Gypsum and Marble ', 'value': 0.01},
       {'label': 'Glass Fiber Wool', 'value': 0.56},
       ], value=0.1, id='Wall_Material'
    ),
    
    html.H4("Flooring Finish", style = {"textAlign": "center"}),
    dcc.Dropdown(options=[
       {'label': 'Vinyl Tile', 'value': 0.03},
       {'label': 'Wood Boards on Concrete', 'value': 0.07},
       {'label': 'Flooring Block', 'value': 0.10},
       {'label': 'Parquet Flooring on Concrete', 'value': 0.07},
       {'label': 'Carpet and Wood on Concrete', 'value': 0.12},
       ], value=0.1, id='Floor_Material'
    ),
    
    html.H4("Hall Geometry Type", style = {"textAlign": "center"}),
    dcc.Dropdown(options=[
       {'label': 'Shoebox', 'value': 1},
       {'label': 'Fan', 'value': 2},
       {'label': 'Oval', 'value': 3},
       {'label': 'Vineyard', 'value': 4},
       {'label': 'Octagon', 'value': 5},
       {'label': 'Hexagon', 'value': 6}, 
       {'label': 'Other', 'value': 7},
       ], value=1, id='Hall_Type'
    ),
    
    html.H2("Estimated Reverberation Time", style = {"textAlign": "center"}),
    
    html.Div(id='slider-output-container')
], style = {"margin": 30})

@app.callback(
    Output('slider-output-container', 'children'),
    Input('Room_Volume', 'value'),
    Input('Surface_Area', 'value'),
    Input('Seating_Capacity', 'value'),
    Input('Noise_Level', 'value'),
    Input('Ceiling_Material', 'value'),
    Input('Wall_Material', 'value'),
    Input('Floor_Material', 'value'),
    Input('Hall_Type', 'value')
)

def Random_Forest_RT_Prediction(Room_Volume, Surface_Area, Seating_Capacity, Noise_Level, Ceiling_Material, Wall_Material, Floor_Material, Hall_Type):
    Design_Vals_test = [Room_Volume, Surface_Area, Seating_Capacity, Noise_Level, Ceiling_Material, Wall_Material, Floor_Material, Hall_Type]
    Design_Vals_test = np.array(Design_Vals_test)
    Design_Vals_test = Design_Vals_test.reshape(-1,8)
    
    RF_RT_pred = rf_RT_best.predict(Design_Vals_test)
    RF_RT_pred = RF_RT_pred[0]
    
    # Sabine EQ
    Abs_Avg = ((Ceiling_Material + Wall_Material + Floor_Material + 0.7) / 4)
    S_RT_pred = (0.161 * Room_Volume) / (Surface_Area * Abs_Avg)
    
    # Norris-Eyring EQ
    NE_RT_pred = (0.161 * Room_Volume) / (-Surface_Area * np.log(1-Abs_Avg))
    
    return 'The Random Forest estimated mid-frequency (500 Hz) RT is {:.3f}'.format(RF_RT_pred) + " seconds. " + 'The Sabine EQ estimated mid-frequency (500 Hz) RT is {:.3f}'.format(S_RT_pred) + " seconds. " + 'The Norris-Eyring EQ estimated mid-frequency (500 Hz) RT is {:.3f}'.format(NE_RT_pred) + " seconds. "


# In[168]:


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

