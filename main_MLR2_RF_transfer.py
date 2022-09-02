#%% Use trained MLR and RF models on training gridcell to predict fine scale ET at recepient gridcell
from operator import index
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error 
import hydroeval as he
import functions as functions
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import joblib
import numpy as np
import random

# input/output directories
load_dir_dataset = "/g/data/w97/sho561/Downscale/BARRA/Training_Testing_new/"
dir_model = "/g/data/w97/sho561/Downscale/BARRA/Models_new/"
pred_eval = "/g/data/w97/sho561/Downscale/BARRA/Prediction_Evaluation_new/"


all_years = list(range(1990,2019))
recepient_grids = [642, 714, 720, 1207,  1233, 1682, 1728, 2348, 2817, 2855, 3002, 3114, 3346, 3809,  4233, 4322, 4615, 4623, 6081, 6145]
featuresList = ['av_lat_hflx', 'av_mslp', 'av_netlwsfc', 'av_netswsfc', 'av_qsair_scrn', 'av_temp_scrn', 
'av_canopy_height', 'av_uwnd10m', 'av_vwnd10m', 'av_leaf_area_index', 'soil_albedo', 'soil_porosity', 'soil_bulk_density', 'topog' ]

# number predictors
npred = len(featuresList)

# I will run the script on 'experiment 1', i.e. training on 10 years, I might try other experiments in the future. 
experiment = 'exp1' 
# number of trees in Random forest
ntrees = 100

# donor and recepient grid cells
donors_df = pd.read_csv(load_dir_dataset + 'representative_grids_transfer_final.csv')
donors_df = donors_df[['receps', 'donors']].drop_duplicates()
#%% predict separately for each recepient grid cell
for recepient_grid in recepient_grids:    

    print(recepient_grid)  
    # extract the donor grid of the current recepient grid
    train_grid =  donors_df[donors_df['receps']==recepient_grid].donors.item()
    
    ## load the trained MLR model
    filename_model = dir_model + "/MLR_%s_%spred_%s.joblib" %(train_grid, npred, experiment)
    mlr_loaded = joblib.load(filename_model)

    # load the RF model 
    filename_rf = dir_model + "/RF_MLR_%s_trees%s_%spred_%s.joblib" %(train_grid, ntrees, npred, experiment)
    rf_loaded = joblib.load(filename_rf)

    
    test_df = pd.DataFrame()
    for year in all_years:    
        # Read yearly files
        filename_dataset = load_dir_dataset +'%s_%s_predictors_target.csv' %(recepient_grid, year)
        single_year_df = pd.read_csv(filename_dataset)
        # MLR and RF don't like Null values
        single_year_df = single_year_df.dropna(axis=0)
        test_df = pd.concat([test_df, single_year_df])           
    
    # keep predictors only
    X_test = test_df[featuresList] 
    # normalise the data before prediction for MLR only (in RF we don't need to normalise the data)
    X_test = StandardScaler().fit_transform(X_test) 
    predictions = mlr_loaded.predict(X_test)


    # add a column for the predicted values
    test_df['MLR_transfer'] = predictions
    # predict the bias
    features_df = test_df[featuresList]
    test_df['bias'] = rf_loaded.predict(features_df)
    test_df = test_df[['ref_fine_cell', 'year', 'month', 'day','target','MLR_transfer','bias']]
    # bias correct
    test_df['MLR_RF_transfer'] = test_df['MLR_transfer'] - test_df['bias']
    # Save the predictions
    filename_test_1990_2018 = pred_eval + "MLR_RF_%s_100trees_pred_transfer_%s_%spred_1990_2018_%s.csv" %(recepient_grid, train_grid, npred, experiment)
    test_df.to_csv(filename_test_1990_2018, index=False)
    


#%%
