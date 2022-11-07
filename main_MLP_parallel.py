## Build a Multi layer perceptron model for each grid to downscale Evapotranspiration dataset by a scale factor of 9, from 12km to 1.5 km
#-----------------------------------------------------------

# Normal libraries:
from operator import index
from random import sample
import time

# Libraries for handling CPU data:
import pandas as pd
import numpy as np

# Libraries for ML + GPU:
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
# Importing Pytorch multiprocessing library
import torch.multiprocessing as mp

# Libraries for testing:
import hydroeval as he

# Importing slef-made libraries:
import config # define the global variables
from mlp_model import MLP_model # This is where I defined the MLP model
from functions import BarraDataset # Convert the data into torch and move it to the GPU
import functions as functions # Functions used for evaluating the model
import train as train # contains the training functions



#-----------------------------------------------------------
# Value needed for torch runs
seed = 100

#-----------------------------------------------------------

# '''' Initialisation ''''
version = config.version # the current version of the analysis. In each version, I tried different hyperparameters for the model and/or different ways for scaling the training/testin data
# List of predictors
featuresList = config.predictors

#-----------------------------------------------------------

# input/output directories
load_dir_dataset = "/scratch/vp91/CLEX/Training_Testing/"
dir_model = "/scratch/vp91/CLEX/Models_accelerated/"
pred_eval = "/scratch/vp91/CLEX/Prediction_Evaluation_accelerated/"

all_years = list(range(1990,2019)) # BARRA dataset has 29 years 1990 - 2018

# Selected coarse gridcells which I want to downscale. Training files are prepared per coarse grid and year
train_grids = [642, 714, 720, 1207,  1233, 1682, 1728] #, 2348] #, 2817, 2855, 3002, 3114, 3346, 3809,  4233, 4322, 4615, 4623, 6081, 6145, 642, 714, 720, 1207,  1233, 1682]#, 1728, 2348, 2817, 2855, 3002, 3114, 3346, 3809,  4233]

# Number of processes that will run in parallel
num_processes =  len(train_grids)

# I will run the script on 'experiment 1', i.e. training on 10 years, I might try other experiments in the future. 
experiment = 'exp1' 
if experiment == 'exp1':
    batch_size = 50
    epoch_number = 5 ##originally 50
    train_years = [1990, 1991, 1992, 1995, 1996, 2001, 2003, 2004, 2016, 2018 ]
    test_years = list(set(all_years) - set(train_years)) 

#-----------------------------------------------------------

# Creating functions to make the code more readable:

# Read in data per year and concatanate all years together.
def file_concat(coarse_grid, y):

    sample_df = pd.concat([pd.read_csv(load_dir_dataset +'%s_%s_predictors_target.csv' %(coarse_grid, year)) for year in y], axis=0).dropna(axis=0)

    return sample_df

def train_predict_evaluate(mlp, trainloader, filename_model, current_grid, scaler):

            
    mlp = train.train_mlp_mp(mlp, trainloader, seed, epoch_number, batch_size, filename_model)

    # Prepare testing dataset
    # read yearly files for all testing years and concatenate them into one big testing dataframe 
    # Call data function:
    test_data = file_concat(current_grid, test_years)
        
    # Keep only the predictors
    X_test = test_data[featuresList]
    #scaler = StandardScaler()
    X_test = scaler.transform(X_test) 
    

    filename_testing = pred_eval + "MLP_%s_transfer_%s_%sepochs_%sbatch_pred_1990_2018_%s_%s.csv" %(current_grid, current_grid, epoch_number, batch_size, experiment, config.version)
    
    # add a column for the predicted values

    test_data['MLP'] = predict(X_test, mlp)
    # Keep only date, target and predicted
    out_sample_df = test_data[['ref_fine_cell', 'year', 'month', 'day','target', 'MLP']]

    # combine predicted data at training and testing data
    out_sample_df.to_csv(filename_testing, index=False)

    # ''''' Evaluation at the testing years, compare predicted fine gridcells with target gridcells
    #evaluate(out_sample_df, coarse_grid)
    

    return 1

def predict(X, mlp):
    # use the model to predict on the training data
    # transfer of data from cpu to GPU and vice versa 
    X_train = torch.from_numpy(X).cuda()
    predictions = mlp(X_train.float()).cpu()
    predictions = predictions.detach().numpy()

    return predictions

def evaluate(data, coarse_grid):
    out_sample_df = data
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

#-----------------------------------------------------------

def main():


    #-----------------------------------------------------------
    # Initialisation - needed for multiprocessing
    trainloader_list = []
    mlp_list = []
    filename_model_list = []
    current_grid_list = []
    scaler_list = []
    
    # Build a new model for each coarse grid in train_grids
    for coarse_grid in train_grids:

        current_grid_list.append(coarse_grid)
        #-----------------------------------------------------------
        # Create the input data:
        in_data = file_concat(coarse_grid, all_years)
    
        #  # Keep only the predictors and normalise the training data
        scaler = StandardScaler()
        X = in_data[featuresList]
        X = scaler.fit_transform(X) 
        Y  = in_data[['target']].to_numpy()

        scaler_list.append(scaler)
        # ----------------------------------------------------------
        # Prepare arguments needed for parallel training
        # convert our dataset to a PyTorch-compatible dataset. Batch and shuffle the dataset first, so that no hidden patterns in data collection can disturb model training. 
        dataset = BarraDataset(X, Y)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        torch.manual_seed(seed)
        # Initialize the MLP
        mlp = MLP_model().cuda() 
        
        # Name of the model
        filename_model =  dir_model + "MLP_%s_epoch%s_batch%s_%s_%s.pth" %( coarse_grid, epoch_number, batch_size, experiment, version) 

        trainloader_list.append(trainloader)
        filename_model_list.append(filename_model)
        mlp_list.append(mlp)


    #-----------------------------------------------------------
    # Multi-process Training the model with the data:
    start_time = time.time()

    mp.set_start_method('spawn', force=True) #added
    mlp.share_memory() #added
    processes = [] #added
    for rank in range(num_processes):
        p = mp.Process(target = train_predict_evaluate, args = (mlp_list[rank], trainloader_list[rank], filename_model_list[rank], current_grid_list[rank], scaler_list[rank])) 
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    training_duration = time.time() - start_time
    print("--- %s minutes ---" % round(training_duration/60, 1))
    #-----------------------------------------------------------
    
    print(num_processes)
    

       
                

        #-----------------------------------------------------------
        # # add a column for the predicted values
        # in_data['MLP'] = predict(X, mlp)

        # # Keep only date, target and predicted. Later, combine this dataframe with the testing predictions and save the combined file
        # in_sample_df = in_data[['ref_fine_cell', 'year', 'month', 'day','target', 'MLP']]

        # #-----------------------------------------------------------

        # ## ''''''''  evaluate the model in the testing years & save predictions for all years 1990 - 2018 in a single file

        # # read yearly files for all testing years and concatenate them into one big testing dataframe 
        # # Call data function:
        # test_data = file_concat(coarse_grid, test_years)

        # # Keep only the predictors
        # X_test = test_data[featuresList]
        # X_test = scaler.transform(X_test) 

        # # add a column for the predicted values
        # test_data['MLP'] = predict(X_test, mlp)

        # # Keep only date, target and predicted
        # out_sample_df = test_data[['ref_fine_cell', 'year', 'month', 'day','target', 'MLP']]

        # # combine predicted data at training and testing data
        # all_sample_df = pd.concat([in_sample_df, out_sample_df], ignore_index=True)
        # filename_test_1990_2018 = pred_eval + "MLP_%s_transfer_%s_%sepochs_%sbatch_pred_1990_2018_%s_%s.csv" %(coarse_grid, coarse_grid, epoch_number, batch_size, experiment, config.version)
        # all_sample_df.to_csv(filename_test_1990_2018, index=False)

        # # ''''' Evaluation at the testing years, compare predicted fine gridcells with target gridcells
        # evaluate(out_sample_df, coarse_grid)

if __name__ == "__main__":
    
    main()
