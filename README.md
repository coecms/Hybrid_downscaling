# ML_downscaling
Sanaa's downscaling climate model using machine learning. Code needs to be optimised for GPUs.

Main scripts:

- main_MLP.py: trains a Multi layer perceptron model for every grid, uses 20 grids only. It uses a GPU version of PyTorch and it needs to run with GPU.

- main_MLP_transfer.py: uses the saved model created in ‘main_MLP.py’ to predict on new data. Uses GPU since it's using PyTorch-GPU.

---------------------------------------------------------------------------

- main_MLR2_RF.py: trains a multivariate linear regression model (Scikit-learn) and Random Forest (Scikit-learn) for every grid, uses 20 grids. Uses CPUs.

- main_MLR2_RF_transfer.py: uses the saved models created in ‘main_MLR2_RF.py’ to predict on new data. Uses CPUs.

---------------------------------------------------------------------------

- The main files call other files config.py, functions.py, mlp_model.py, and train.py.

In the future more than 6000 grids will be needed. 

The script reads data from

```
/g/data/w97/sho561/Downscale/BARRA/Training_Testing_new/
```
And create new files in

```
/g/data/w97/sho561/Downscale/BARRA/Models_new/
/g/data/w97/sho561/Downscale/BARRA/Prediction_Evaluation_new/
```

Python Modules needed:
- torch
- pandas
- sklearn
- hydroeval
- numpy
- operator