# Hybrid Downscaling
Downscaling climate model using GPU-enabled machine learning.

Updated: 07/11/22

# Paper info:
This code is used for the following paper:

Sanaa Hobeichi, Nidhi Nishant, Yawen Shao, Gab Abramowitz, Andy Pitman, Steve Sherwood, Craig Bishop and Samuel Green. Using Machine Learning to Cut the Cost of Dynamical Downscaling. Accepted in Earth's Future, 2023

# Code:
Two machine learning methods were used to approach this task. The first method uses a Multi layer perceptron (MLP) model using pytorch. The second method uses a multivariate linear regression model and Random Forest (MLR_RF) using Scikit-learn.

The serial version of both methods work fine and have comparable results, they're just slow to run. The pytorch method has received the most recent development in terms of speeding-up but the scikit-learn method will be worked on in the future using RAPIDs. 


## Running MLP in parallel
---------------------------------------------------------------------------
The MLP model has been improved to run in parallel on 1 GPU (muliptle GPUs a work-in-progress), this allows 4 grids to run at the same time allowing for a x4 increase in speed. 

main_MLP_parallel.py is the main code for this and can be run as normal:
```bash
python3 main_MLP_parallel
```

The main file calls other files config.py, functions.py, mlp_model.py, and train.py.

Python Modules needed:
- torch
- pandas
- numpy
- operator
- hydroeval

The script takes advantage of the ```torch.multiprocessing``` library to parallelise the loop over multiple grid cells.

The script reads data from

```
/g/data/w97/sho561/Downscale/BARRA/Training_Testing_new/
```
And creates new files in

```
/g/data/w97/sho561/Downscale/BARRA/Models_new/
/g/data/w97/sho561/Downscale/BARRA/Prediction_Evaluation_new/
```

## Running MLP in serial
---------------------------------------------------------------------------

Main scripts:

- main_MLP_serial.py: trains a Multi layer perceptron model for every grid, uses 20 grids only. It uses a GPU version of PyTorch and it needs to run with GPU.

- main_MLP_transfer.py: uses the saved model created in ‘main_MLP.py’ to predict on new data. Uses GPU since it's using PyTorch-GPU.



- main_MLR2_RF.py: trains a multivariate linear regression model (Scikit-learn) and Random Forest (Scikit-learn) for every grid, uses 20 grids. Uses CPUs.

- main_MLR2_RF_transfer.py: uses the saved models created in ‘main_MLR2_RF.py’ to predict on new data. Uses CPUs.

---------------------------------------------------------------------------
