#!/usr/bin/env python
# coding: utf-8

# In[1]:


# By Jonathan Broyles (Fall 2021 - Spring 2023)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl as pyxl
import seaborn as sns
import matplotlib.patches as mpatches
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


plt.figure(figsize=(8,8))

patches, texts, autotexts  = plt.pie(
    x=Hall_numbers, 
    labels=Hall_labels,
    # show percentage with two decimal points
    autopct='%1.0f%%',
    # increase the size of all text elements
    textprops={'fontsize':14},
    startangle=180,
    counterclock=False,
    colors = sns.color_palette('Spectral_r', 7),
    explode=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
#     labeldistance=0.85,
    # Distance of percent labels from the center
#     pctdistance=0.6
)

plt.rcParams['font.sans-serif'] = "Arial" # Could do Times New Roman, others...
plt.rcParams['font.family'] = "Arial" # Could do Times New Roman, others...

# # Customize text labels
# for text in texts:
#     text.set_horizontalalignment('center')

texts[0].set_fontsize(20)
texts[1].set_fontsize(20)
texts[2].set_fontsize(20)
texts[3].set_fontsize(20)
texts[4].set_fontsize(20)
texts[5].set_fontsize(20)
texts[6].set_fontsize(20)

# Customize percent labels
for autotext in autotexts:
    autotext.set_horizontalalignment('center')
    autotext.set_fontstyle('italic')

# # Add Title 
# plt.title(
#     label="Population Distribution By Age Groups", 
#     fontdict={"fontsize":16},
#     pad=20
# )

plt.savefig("Concert Hall Type Breakdown.png", dpi = 1200)
plt.show()


# # Random Forest Model

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier


# In[6]:


df


# In[7]:


df_clean_RF_x = df[["Seating Capacity", "Room Volume m^3", "Surface Area m^2", "Noise Level Number", "Ceiling Alpha", "Wall Alpha","Floor Alpha", "Hall Type Number"]] #, "Stage Area m^2", "Noise Level Number"
df_clean_RF_y = df["RT - 500 Hz Unocc"]


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(df_clean_RF_x, df_clean_RF_y, test_size = 0.3, random_state=1)


# In[9]:


rf_RT = RandomForestRegressor()


# In[10]:


from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3, 4, 5, 6],
    'min_samples_leaf': [3, 4, 5, 6],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 400, 500, 1000, 2500, 5000]
}


# In[11]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf_RT, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)


# In[12]:


# Fit the grid search to the data
grid_search.fit(X_train.values, y_train)
grid_search.best_params_

%%time


# In[16]:


grid_search.best_params_


# In[13]:


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test.values)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} seconds.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train.values, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)


# In[14]:


best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)


# In[15]:


print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


# In[75]:


rf_RT_best = RandomForestRegressor(bootstrap = True, max_depth = 100, max_features = 3, min_samples_leaf = 5, min_samples_split = 12,
                                   n_estimators = 500)
rf_RT_best.fit(X_train.values, y_train)


# In[76]:


y_pred = rf_RT_best.predict(X_test.values)


# In[77]:


headers = ["Seating Capacity", "Room Volume", "Room Surface Area", "NC-Level", r"Ceiling $\alpha$",  r"Wall $\alpha$",  r"Floor $\alpha$", "Hall Type"]
#headers = list(df_clean_RF_x.columns)

fig = plt.figure(figsize = (9, 7))
sorted_idx = rf_RT_best.feature_importances_.argsort()

imps = rf_RT_best.feature_importances_[sorted_idx]
my_cmap = plt.get_cmap("plasma_r")
rescale = lambda imps: (imps - np.min(imps)) / (np.max(imps) - np.min(imps))

plt.barh(np.array(headers)[sorted_idx], rf_RT_best.feature_importances_[sorted_idx], color=my_cmap(rescale(imps)), edgecolor = "dimgray", alpha = 0.85)

#plt.barh(np.array(headers), rf.feature_importances_)

plt.xlabel("Random Forest Feature Importance")
plt.title("Parameter Importance for Predicting RT-500")

plt.rcParams['font.sans-serif'] = "Arial" # Could do Times New Roman, others...
plt.rcParams['font.family'] = "Arial" # Could do Times New Roman, others...

plt.savefig("Concert Hall Predicting RT - Parameter Importance using Random Forests Final.pdf", dpi = 1200, bbox_inches='tight', pad_inches=0.5,)
plt.show()


# In[78]:


r2_score(y_test, y_pred)


# In[79]:


rf_RT_best.score(X_train.values, y_train)


# In[80]:


# Performance metrics
errors = abs(y_pred - y_test)
print('Metrics for Random Forest Trained on Original Data')
print('Average RT absolute error:', round(np.mean(errors), 2), 'seconds.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

RMSE = mean_squared_error(y_test, y_pred, squared = False)
NRMSE = RMSE / (max(df["RT - 500 Hz Unocc"]) - min(df["RT - 500 Hz Unocc"]))
print('RMSE:', round(RMSE, 2))
print('NRMSE:', round(NRMSE, 2))


# # Compare the RT-Regression Model to Other Analytical EQs

# In[81]:


Reported_RT = df_clean_RF_x


# In[82]:


Sabine_RT = df["Sabine RT Calc"]
Norris_Eyring_RT = df["Norris-Eyring EQ"]


# In[83]:


RF_preds = rf_RT_best.predict(Reported_RT)
num_halls = range(1,(len(RF_preds)+1))

RF_Perc_Errs = []
for i in range(len(RF_preds)):
    dummy = abs((df_clean_RF_y[i] - RF_preds[i]) / df_clean_RF_y[i]) * 100
    RF_Perc_Errs.append(dummy)


# In[84]:


plt.scatter(num_halls, df_clean_RF_y, color = "blue", label = "Reported RT", zorder = 8)
plt.scatter(num_halls, RF_preds, color = "orange", label = "Random Forest")
plt.scatter(num_halls, Sabine_RT, color = "peru", label = "Sabine EQ")
plt.scatter(num_halls, Norris_Eyring_RT, color = "yellowgreen", label = "Norris-Eyring EQ")

plt.ylim([0.5,3.5])
plt.rcParams['font.sans-serif'] = "Arial" # Could do Times New Roman, others...
plt.rcParams['font.family'] = "Arial" # Could do Times New Roman, others...

plt.ylabel("Mid-Frequency Reverberation Time", fontsize = 12)

plt.legend()
plt.show


# In[85]:


Sabine_Perc_Errs = df["Percent Error 1"]
Norris_Eyring_Perc_Errs = df["Percent Error 2"]


# In[86]:


RF_Perc_Errs = np.array(RF_Perc_Errs)
print("RF Average Error = " + str(np.mean(RF_Perc_Errs)) + " %")

Sabine_Perc_Errs = np.array(Sabine_Perc_Errs)
print("Sabine EQ Average Error = " + str(np.mean(Sabine_Perc_Errs)) + " %")

Norris_Eyring_Perc_Errs = np.array(Norris_Eyring_Perc_Errs)
print("Norris Eyring EQ Average Error = " + str(np.mean(Norris_Eyring_Perc_Errs)) + " %")


# In[87]:


x = np.arange(50)


# In[88]:


num_halls = np.array(num_halls)


# In[89]:


# plot data in grouped manner of bar type
fig = plt.figure(figsize = (10, 6))

colors = ['#66b3ff','#ffcc99','#99ff99']

width = 0.2
plt.bar(num_halls-0.2, RF_Perc_Errs, width, color = "#66b3ff", edgecolor = "dimgray", label = "Random Forest", alpha = 1, zorder = 8)
plt.bar(num_halls, Sabine_Perc_Errs, width, color = "#ffcc99", edgecolor = "dimgray", label = "Sabine EQ", alpha = 0.8)
plt.bar(num_halls+0.2, Norris_Eyring_Perc_Errs, width, color = "#99ff99", edgecolor = "dimgray", label = "Norris-Eyring EQ", alpha = 0.8)


plt.ylim([0,70])
plt.rcParams['font.sans-serif'] = "Arial" # Could do Times New Roman, others...
plt.rcParams['font.family'] = "Arial" # Could do Times New Roman, others...

plt.xticks([])
plt.xlabel("Concert Hall", fontsize = 12)
plt.ylabel("Percent Error", fontsize = 12)
plt.legend(loc = "upper left", fontsize = 14, edgecolor = "white")

plt.savefig("Percent Errors Final.pdf", dpi = 1200, bbox_inches='tight', pad_inches=0.5,)
plt.show()


# In[102]:


fig = plt.figure(figsize = (10, 6))

plt.scatter(num_halls, RF_Perc_Errs, marker = "d", color = "mediumseagreen", s = 70, label = "Random Forest", zorder = 8)
plt.scatter(num_halls, Sabine_Perc_Errs, marker = "o", color = "steelblue", s = 55, label = "Sabines EQ", zorder = 4)
plt.scatter(num_halls, Norris_Eyring_Perc_Errs, marker = "s", color = "goldenrod", s = 50, label = "Norris-Eyring EQ", zorder= 2)

plt.vlines(x = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,
               21.5,22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5,30.5,31.5,32.5,33.5,34.5,35.5,36.5,37.5,38.5,39.5,
               40.5,41.5,42.5,43.5,44.5,45.5,46.5,47.5,48.5,49.5,50.5], ymin = -3, ymax = 70, color = "gray", linewidth = 0.33)

plt.hlines(y = 20, xmin = -1, xmax = 52, color = "gray", linewidth = 0.33, zorder = 1)

plt.xlim([-1,52])
plt.ylim([-3,70])
plt.gca().invert_yaxis()


plt.rcParams['font.sans-serif'] = "Arial" # Could do Times New Roman, others...
plt.rcParams['font.family'] = "Arial" # Could do Times New Roman, others...

plt.xticks([])
plt.xlabel("Concert Hall", fontsize = 12)
plt.ylabel("Percent Error (%)", fontsize = 12)
plt.legend(loc = "lower left", fontsize = 14, edgecolor = "dimgray", facecolor = "white")

plt.savefig("Percent Errors Scatterplot.pdf", dpi = 1200, bbox_inches='tight', pad_inches=0.5,)
plt.show()


# In[91]:


best_model = []
labells = []

for i in range(len(RF_Perc_Errs)):
    if min(RF_Perc_Errs[i], Sabine_Perc_Errs[i], Norris_Eyring_Perc_Errs[i]) == RF_Perc_Errs[i]:
        dummy = 0
        lab = "Random Forest Model"
    elif min(RF_Perc_Errs[i], Sabine_Perc_Errs[i], Norris_Eyring_Perc_Errs[i]) == Sabine_Perc_Errs[i]:
        dummy = 1
        lab = "Sabine EQ"
    elif min(RF_Perc_Errs[i], Sabine_Perc_Errs[i], Norris_Eyring_Perc_Errs[i]) == Norris_Eyring_Perc_Errs[i]:
        dummy = 2
        lab = "Norris-Eyring EQ"
    print(str(i+1) + ": " + str(dummy))
    best_model.append(dummy)
    labells.append(lab)


# In[103]:


best_modell = [25, 16, 9]


# In[105]:


labels = ["Random Forest", "Sabine EQ", "Norris-Eyring EQ"]

my_explode = (0.02, 0.01, 0.01)
colors = ['mediumseagreen','steelblue','goldenrod']

# Creating plot
fig = plt.figure(figsize =(10, 7))
patches, texts, pcts = plt.pie(best_modell, labels = labels, startangle = 90, autopct='%1.0f%%',  colors = colors, explode=my_explode, textprops = {"fontsize": 13}, wedgeprops = {"linewidth": 1, "edgecolor": "white"})
#plt.title("The best RT prediction is obtained by:", fontsize = 13)
plt.setp(pcts, color='white', fontweight='bold', fontsize = "xx-large")
plt.rcParams['font.sans-serif'] = "Arial" # Could do Times New Roman, others...
plt.rcParams['font.family'] = "Arial" # Could do Times New Roman, others...

# show plot
plt.savefig("Best Prediction Pie Chart.pdf", dpi = 1200, bbox_inches='tight', pad_inches=0.5,)
plt.show()


# # Deployment of the RF Model as an Interactive Design Tool

# In[111]:


from dash import Dash, dcc, html, Input, Output
import plotly.express as px


# In[113]:


import dash_bootstrap_components as dbc


# In[167]:


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(external_stylesheets=[dbc.themes.YETI]) # , external_stylesheets=external_stylesheets

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

