## Build a Multi layer perceptron model for each grid to downscale Evapotranspiration dataset by a scale factor of 9, from 12km to 1.5 km

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

# '''' Initialisation ''''
version = config.version # the current version of the analysis. In each version, I tried different hyperparameters for the model and/or different ways for scaling the training/testin data
# List of predictors
featuresList = config.predictors

all_years = list(range(1990,2019)) # BARRA dataset has 29 years 1990 - 2018

# Selected coarse gridcells which I want to downscale. Training files are prepared per coarse grid and year
train_grids = [642, 714, 720, 1207,  1233, 1682, 1728, 2348, 2817, 2855, 3002, 3114, 3346, 3809,  4233, 4322, 4615, 4623, 6081, 6145]

# I will run the script on 'experiment 1', i.e. training on 10 years, I might try other experiments in the future. 
experiment = 'exp1' 
if experiment == 'exp1':
    batch_size = 50
    epoch_number = 50
    train_years = [1990, 1991, 1992, 1995, 1996, 2001, 2003, 2004, 2016, 2018 ]
    test_years = list(set(all_years) - set(train_years)) 

#%%
# Build a new model for each coarse grid in train_grids
for coarse_grid in train_grids:
    print(coarse_grid)

    # initialisation, dataframe containing all the training samples
    in_sample_df = pd.DataFrame()
    
    # read yearly files for all training years and concatenate them into one big training dataframe 
    for year in train_years:
       
        filename_dataset = load_dir_dataset +'%s_%s_predictors_target.csv' %(coarse_grid, year)
        single_year_df = pd.read_csv(filename_dataset)
        # Multi layer perceptron doesn't like Null values
        single_year_df = single_year_df.dropna(axis=0)
        in_sample_df = pd.concat([in_sample_df, single_year_df])

    #  # Keep only the predictors and normalise the training data
    scaler = StandardScaler()
    X = in_sample_df[featuresList]
    X = scaler.fit_transform(X) 
    y  = in_sample_df[['target']].to_numpy()

    
    print('shape of X is ', X.shape)

    
    torch.manual_seed(seed)
    # convert our dataset to a PyTorch-compatible dataset. Batch and shuffle the dataset first, so that no hidden patterns in data collection can disturb model training. 
    dataset = BarraDataset(X, y)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # This is where I am going to save the output model
    filename_model =  dir_model + "MLP_%s_epoch%s_batch%s_%s_%s.pth" %( coarse_grid, epoch_number, batch_size, experiment, version)
    
   
    torch.manual_seed(seed)
    # Initialize the MLP
    mlp = MLP_model().cuda() 
    # Define the loss function and optimizer  
    loss_function = nn.MSELoss() # alternatively I can choose the nn.L1Loss(), or any pre-defined custom function trial and error will tell me the optimal loss function
    #loss_function = functions.custom_loss_function

    ## choose ADAM otimiser (standard) with common learning rate equal to 1e-4
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    
    # train the model and track the time needed for training
    start_time = time.time()
    mlp = train.train_mlp(mlp, trainloader, optimizer, loss_function, seed, epoch_number, batch_size)
    #mlp = torch.load(filename_model), later, I can load the saved model using torch.load
    
    # print training time
    print('number of epochs is %s' % epoch_number)
    training_duration = time.time() - start_time
    print("--- %s minutes ---" % round(training_duration/60, 1))

    # save the model 
    torch.save(mlp, filename_model)

    # use the model to predict on the training data, do necessary transfer of data from cpu to GPU and vice versa 
    X_train = torch.from_numpy(X).cuda()
    predictions = mlp(X_train.float()).cpu()
    predictions = predictions.detach().numpy()
    # add a column for the predicted values
    in_sample_df['MLP'] = predictions

     # Keep only date, target and predicted. Later, combine this dataframe with the testing predictions and save the combined file
    in_sample_df = in_sample_df[['ref_fine_cell', 'year', 'month', 'day','target', 'MLP']]

    
    ## ''''''''  evaluate the model in the testing years & save predictions for all years 1990 - 2018 in a single file
    
    # read yearly files for all training years and concatenate them into one big training dataframe 
    out_sample_df = pd.DataFrame()
    
    for year in test_years:

        filename_dataset = load_dir_dataset +'%s_%s_predictors_target.csv' %(coarse_grid, year)
        single_year_df = pd.read_csv(filename_dataset)
        # Multi layer perceptron doesn't like Null values
        single_year_df = single_year_df.dropna(axis=0)
        out_sample_df = pd.concat([out_sample_df, single_year_df])
    
    # Keep only the predictors
    X_test = out_sample_df[featuresList]
    X_test = scaler.transform(X_test) 

    # make prediction, do necessary transfer of data from/to GPU and CPU
    print("predicting ...")
    X_test = torch.from_numpy(X_test).cuda() # to GPU
    predictions = mlp(X_test.float()).cpu() # to CPU
    predictions = predictions.detach().numpy()
    # add a column for the predicted values
    out_sample_df['MLP'] = predictions

    # Keep only date, target and predicted
    out_sample_df = out_sample_df[['ref_fine_cell', 'year', 'month', 'day','target', 'MLP']]

    # combine predicted data at training and testing data
    all_sample_df = pd.concat([in_sample_df, out_sample_df], ignore_index=True)
    
    filename_test_1990_2018 = pred_eval + "MLP_%s_transfer_%s_%sepochs_%sbatch_pred_1990_2018_%s_%s.csv" %(coarse_grid, coarse_grid, epoch_number, batch_size, experiment, config.version)

    all_sample_df.to_csv(filename_test_1990_2018, index=False)


    # ''''' Evaluation at the testing years, compare predicted fine gridcells with target gridcells
    fine_grid_cells = out_sample_df['ref_fine_cell'].unique().tolist()   
   
    all_mean_eval_df = pd.DataFrame()
    for test_year in test_years:
        all_eval_df = pd.DataFrame()
        for fine_grid in fine_grid_cells:
            subset_test_df = out_sample_df[(out_sample_df['ref_fine_cell'] == fine_grid) & (out_sample_df['year'] == test_year)]
            eval_data = functions.testing_performance(subset_test_df, 'MLP', test_year)
            eval_df = pd.DataFrame(eval_data)
            eval_df['ref_fine_cell'] = fine_grid
            all_eval_df = pd.concat([all_eval_df, eval_df])

        #all_test_df = pd.concat([all_test_df, all_eval_df])
        mean_eval_df = pd.DataFrame(all_eval_df.mean(axis=0)).T
        mean_eval_df = mean_eval_df.drop(['ref_fine_cell'], axis=1)
        all_mean_eval_df = pd.concat([all_mean_eval_df, mean_eval_df])

    # save the results of evaluation 
    filename_evaluation_mean =  pred_eval + 'MLP_%s_transform_mean_testing_perf_trained_on_%sgrid_epoch%s_batch%s_%s_%s.csv' %(coarse_grid, coarse_grid, epoch_number, batch_size, experiment, version)
    all_mean_eval_df.to_csv(filename_evaluation_mean, index=False)
