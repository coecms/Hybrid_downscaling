#%% Use trained models on training gridcell to predict fine scale ET at recepient gridcell
'''
This is the main file
with help from https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-create-a-neural-network-for-regression-with-pytorch.md
'''

from operator import index
from random import sample
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error 
import hydroeval as he
import config # define the global variables
from mlp_model import MLP_model # This is where I defined the MLP model
from functions import BarraDataset # Convert the data into torch and move it to the GPU
import functions as functions # Functions used for evaluating the model
import train as train # contains the training functions
import time
import numpy as np
seed = 100

# input/output directories
load_dir_dataset = "/g/data/w97/sho561/Downscale/BARRA/Training_Testing_new/"
dir_model = "/g/data/w97/sho561/Downscale/BARRA/Models_new/"
pred_eval = "/g/data/w97/sho561/Downscale/BARRA/Prediction_Evaluation_new/"

featuresList = config.predictors 
all_years = list(range(1990,2019))

# '''' Initialisation ''''
version = config.version # the current version of the analysis. In each version, I tried different hyperparameters for the model and/or different ways for scaling the training/testin data
# List of predictors
featuresList = config.predictors

all_years = list(range(1990,2019)) # BARRA dataset has 29 years 1990 - 2018

# Selected coarse gridcells which I want to downscale. Training files are prepared per coarse grid and year
#recepient_grids = [642, 714, 720, 1207,  1233, 1682, 1728, 2348, 2817, 2855, 3002, 3114, 3346, 3809,  4233, 4322, 4615, 4623, 6081, 6145]
recepient_grids = [3114, 3346, 3809,  4233, 4322, 4615, 4623, 6081, 6145]


# I will run the script on 'experiment 1', i.e. training on 10 years, I might try other experiments in the future. 
experiment = 'exp1' 
if experiment == 'exp1':
    batch_size = 50
    epoch_number = 50

# donor and recepient grid cells
donors_df = pd.read_csv(load_dir_dataset + 'representative_grids_transfer_final.csv')
donors_df = donors_df[['receps', 'donors']].drop_duplicates()

print(' --- lets start -----')
#%% predict separately for each recepient grid cell
for recepient_grid in recepient_grids:
    
    print(recepient_grid)  
    # extract the donor grid of the current recepient grid
    train_grid =  donors_df[donors_df['receps']==recepient_grid].donors.item()
    print(train_grid)      
    # load the trained MLP model 
    filename_model =  dir_model + "MLP_%s_epoch%s_batch%s_%s_%s.pth" %(train_grid, epoch_number, batch_size, experiment, version)
    print(filename_model)
    mlp = torch.load(filename_model)

    # load data of recepient grid cell and stack all the yearly files in one large dataframe
    all_sample_df = pd.DataFrame()
    for year in all_years:
        # Read yearly files
        filename_dataset = load_dir_dataset +'%s_%s_predictors_target.csv' %(recepient_grid, year)
        single_year_df = pd.read_csv(filename_dataset)
        # Multi layer perceptron doesn't like Null values
        single_year_df = single_year_df.dropna(axis=0)
        all_sample_df = pd.concat([all_sample_df, single_year_df])


    # keep predictors only
    X_test = all_sample_df[featuresList]
    # normalise the data before prediction
    X_test = StandardScaler().fit_transform(X_test) 
    
    # make prediction, transfer from/to CPU and GPU as needed
    print("predicting ...")
    X_test = torch.from_numpy(X_test).cuda()
    predictions = mlp(X_test.float()).cpu()
    predictions = predictions.detach().numpy()
    # add a column for the predicted values
    all_sample_df['MLP'] = predictions

    # Keep only date, target and prediction
    all_sample_df = all_sample_df[['ref_fine_cell', 'year', 'month', 'day','target', 'MLP']]

    # save predictions for all years 
    filename_test_1990_2018 = pred_eval + "MLP_%s_transfer_%s_%sepochs_%sbatch_pred_1990_2018_%s_%s.csv" %(recepient_grid, train_grid, epoch_number, batch_size, experiment, config.version)
    all_sample_df.to_csv(filename_test_1990_2018, index=False)


    # evaluate separately for each fine grid cell and year
    print("evaluating ...")
    fine_grid_cells = all_sample_df['ref_fine_cell'].unique().tolist()
    all_mean_eval_df = pd.DataFrame()

    for test_year in all_years:
        all_eval_df = pd.DataFrame()
        for fine_grid in fine_grid_cells:
            subset_test_df = all_sample_df[(all_sample_df['ref_fine_cell'] == fine_grid) & (all_sample_df['year'] == test_year)]
            eval_data = functions.testing_performance(subset_test_df, 'MLP', test_year)
            eval_df = pd.DataFrame(eval_data)
            eval_df['ref_fine_cell'] = fine_grid

            # combine yearly performance in a single file
            all_eval_df = pd.concat([all_eval_df, eval_df])

    
        mean_eval_df = pd.DataFrame(all_eval_df.mean(axis=0)).T
        mean_eval_df = mean_eval_df.drop(['ref_fine_cell'], axis=1)
        all_mean_eval_df = pd.concat([all_mean_eval_df, mean_eval_df])

    ## save evaluation results
    filename_evaluation_mean =  pred_eval + 'MLP_%s_mean_testing_perf_trained_on_%s_cluster_epoch%s_batch%s_%s_%s.csv' %(recepient_grid, train_grid, epoch_number, batch_size, experiment, version)
    all_mean_eval_df.to_csv(filename_evaluation_mean, index=False)

    print("##############  END #################")


# 
# %%
