## Build a Multivariate linear regression model and a bias correction Random Forest model for each grid to downscale Evapotranspiration dataset by a scale factor of 9, from 12km to 1.5 km
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
seed = 100

# input/output directories
load_dir_dataset = "/g/data/w97/sho561/Downscale/BARRA/Training_Testing_new/"
dir_model = "/g/data/w97/sho561/Downscale/BARRA/Models_new/"
pred_eval = "/g/data/w97/sho561/Downscale/BARRA/Prediction_Evaluation_new/"

# Initialisation
all_years = list(range(1990,2019))
train_grids = [642, 714, 720, 1207,  1233, 1682, 1728, 2348, 2817, 2855, 3002, 3114, 3346, 3809,  4233, 4322, 4615, 4623, 6081, 6145]
featuresList = ['av_lat_hflx', 'av_mslp', 'av_netlwsfc', 'av_netswsfc', 'av_qsair_scrn', 'av_temp_scrn', 
'av_canopy_height', 'av_uwnd10m', 'av_vwnd10m', 'av_leaf_area_index', 'soil_albedo', 'soil_porosity', 'soil_bulk_density', 'topog' ]

# number predictors
npred = len(featuresList)

# I will run the script on 'experiment 1', i.e. training on 10 years, I might try other experiments in the future. 
experiment = 'exp1' 
if experiment == 'exp1':
    train_years = [1990, 1991, 1992, 1995, 1996, 2001, 2003, 2004, 2016, 2018 ]
    test_years = list(set(all_years) - set(train_years)) 

# number of trees in Random forest
ntrees = 100

#%%
# Build a new model for each coarse grid in train_grids
for coarse_grid in train_grids:
    print(coarse_grid)


    # training data
    all_sample_df = pd.DataFrame()
    for year in all_years:
        
        filename_dataset = load_dir_dataset +'%s_%s_predictors_target.csv' %(coarse_grid, year)
        single_year_df = pd.read_csv(filename_dataset)
        # Multi layer perceptron doesn't like Null values
        single_year_df = single_year_df.dropna(axis=0)
        all_sample_df = pd.concat([all_sample_df, single_year_df])
        
    
    # concatening will mess up with the index of the combined dataframe
    all_sample_df= all_sample_df.reset_index()

    # split the data into training and testing
    index_testing = np.where(all_sample_df['year'].isin(test_years))
    index_training = np.where(all_sample_df['year'].isin(train_years))
    in_sample_df = all_sample_df.iloc[index_training]
    out_sample_df = all_sample_df.iloc[index_testing]

    # Keep only the predictors and normalise the training data
    X = all_sample_df[featuresList]
    X = StandardScaler().fit_transform(X) 
    X = X[index_training]
    print('shape of X is ', X.shape)

    # target
    y = all_sample_df.iloc[index_training]
    y  = y[['target']].to_numpy()

    # build a multivariate linear regression on the training data
    random.seed(seed)
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
   
    #Save the trained model
    filename_mlr = dir_model + "MLR_%s_%spred_%s.joblib" %(coarse_grid, npred, experiment) 
    joblib.dump(regr, filename_mlr, compress=3)  # compression is ON
    
    #regr = joblib.load(filename_mlr)
    # predict on the training data
    in_sample_df['MLR'] = regr.predict(X)

    print('build a bias correction model with Random Forest')
    # calculate the bias of the predictions, predicted minus target
    in_sample_df['bias'] = in_sample_df['MLR'] -  in_sample_df['target']

    # save for later
    in_sample_df_save = in_sample_df
    # to train random forest, set a threshold on the bias of the training data
    in_sample_df =  in_sample_df.drop(in_sample_df[(in_sample_df.bias < -50) | (in_sample_df.bias > 50)].index)
    # training data for Random Forest
    train_features = in_sample_df[featuresList]
    # target data for Random Forest
    train_labels = in_sample_df[['bias']]

    # initialisation, use all the available processors
    rf_model =  RandomForestRegressor(n_estimators = ntrees, random_state = seed, n_jobs=-1, oob_score=True)
    
    # keep track of the time needed to train Random Forest
    start_time = time.time()
    
    # training
    rf_model.fit(train_features, train_labels)
    training_duration = time.time() - start_time
    print("--- %s minutes ---" % round(training_duration/60, 1))

    # save the model, saving the model is time consuming
    filename_rf = dir_model + "RF_MLR_%s_trees%s_%spred_%s.joblib" %(coarse_grid, ntrees, npred, experiment) ## copied to E:\Data\BARRA_downscale\Final\models
    joblib.dump(rf_model, filename_rf, compress=6)  # compression is ON!

    ## predict and evaluate MLR and RF at the testing data
    all_test_df = pd.DataFrame()
    all_mean_eval_df = pd.DataFrame()
    
    # normalise the entire data first, then keep the testing data
    X_test = all_sample_df[featuresList]
    X_test = StandardScaler().fit_transform(X_test) 
    X_test = X_test[index_testing]

    # make prediction
    print("predicting ...")
    # predict using the trained MLR, add a column for the predicted values
    out_sample_df['MLR'] = regr.predict(X_test)

    # predict the bias using the trained Random Forset
    features_df = out_sample_df[featuresList] 
    out_sample_df['bias'] = rf_model.predict(features_df)
    out_sample_df['MLR_RF'] = out_sample_df['MLR'] - out_sample_df['bias']

    # Save the output at the training and testing data in one single file 
    out_sample_df = out_sample_df[['ref_fine_cell', 'year', 'month', 'day','target', 'MLR_RF', 'MLR', 'bias']]
    fine_grid_cells = out_sample_df['ref_fine_cell'].unique().tolist()

    in_sample_df_save['bias'] = rf_model.predict(in_sample_df_save[featuresList])
    in_sample_df_save['MLR_RF'] = in_sample_df_save['MLR'] - in_sample_df_save['bias']
    training_testing_df = pd.concat([in_sample_df_save[['ref_fine_cell', 'year', 'month', 'day','target', 'MLR_RF', 'MLR', 'bias']], out_sample_df])
    filename_test_1990_2018 = pred_eval + "MLR_RF_%s_100trees_pred_transfer_%s_%spred_1990_2018_%s.csv" %(coarse_grid, coarse_grid, npred, experiment)
    training_testing_df.to_csv(filename_test_1990_2018, index=False)



    # ''''' Evaluation at the testing years, compare predicted fine gridcells with target gridcells
    
    all_mean_eval_df = pd.DataFrame() # dataframe for the performance of MLR
    all_mean_eval_df2 = pd.DataFrame() # dataframe for the performance of MLR followed by RF
    for test_year in test_years:
        all_eval_df = pd.DataFrame()
        all_eval_df2 = pd.DataFrame()
        for fine_grid in fine_grid_cells:
            subset_test_df = out_sample_df[(out_sample_df['ref_fine_cell'] == fine_grid) & (out_sample_df['year'] == test_year)]
            eval_data = functions.testing_performance(subset_test_df, 'MLR_RF', test_year)
            eval_df = pd.DataFrame(eval_data)
            eval_df['ref_fine_cell'] = fine_grid
            all_eval_df = pd.concat([all_eval_df, eval_df])

            eval_data2 = functions.testing_performance(subset_test_df, 'MLR', test_year)
            eval_df2 = pd.DataFrame(eval_data2)
            eval_df2['ref_fine_cell'] = fine_grid
            all_eval_df2 = pd.concat([all_eval_df2, eval_df2])


        mean_eval_df = pd.DataFrame(all_eval_df.mean(axis=0)).T
        mean_eval_df = mean_eval_df.drop(['ref_fine_cell'], axis=1)
        all_mean_eval_df = pd.concat([all_mean_eval_df, mean_eval_df])

        mean_eval_df2 = pd.DataFrame(all_eval_df2.mean(axis=0)).T
        mean_eval_df2 = mean_eval_df2.drop(['ref_fine_cell'], axis=1)
        all_mean_eval_df2 = pd.concat([all_mean_eval_df2, mean_eval_df2])

    # save the results of evaluation 
    filename_evaluation_mean = pred_eval +'MLR_RF_%s_mean_testing_perf_trained_on_%sgrid_%spred_%s_%s.csv' %(coarse_grid, coarse_grid, npred, ntrees, experiment)
    filename_evaluation_mean2 = pred_eval +'MLR_%s_mean_testing_perf_trained_on_%sgrid_%spred_%s_%s.csv' %(coarse_grid, coarse_grid, npred, ntrees, experiment)

    all_mean_eval_df.to_csv(filename_evaluation_mean, index=False)
    all_mean_eval_df2.to_csv(filename_evaluation_mean2, index=False)

    print("##############  END #################")


