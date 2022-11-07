
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader

## 
def train_mlp(mlp, trainloader, optimizer, loss_function, seed, epoch_number, batch_size):
  
    # Set fixed random number seed
    torch.manual_seed(seed)
    # Run the training loop
    #iterate over the entire training dataset for a fixed number of epochs
    for epoch in range(0, epoch_number): # 5 epochs at maximum

    # Print epoch
        print(f'Starting epoch {epoch+1}')
 
        # Iterate over the DataLoader for training data (iterate over all the batches)
        for i, data in enumerate(trainloader, 0):
            
            # Get and prepare inputs
            inputs, targets = data
            # perform some conversions (e.g. Floating point conversion and reshaping) on the inputs and targets in the current batch
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            # Zero the gradients: knowledge of previous improvements (especially important in batch > 0 for every epoch) is no longer available
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs).cuda()
            
            # Compute loss
            loss = loss_function(outputs, targets)
        
            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()


    # Process is complete.
    print('Training process has finished.')
    print('number of epochs ', epoch+1 )


    return mlp

#%%
def train_mlp_mp(mlp, trainloader, seed, epoch_number, batch_size, filename_model):
  
    # Set fixed random number seed
    torch.manual_seed(seed)

    loss_function = nn.MSELoss() # alternatively I can choose the nn.L1Loss(), or any pre-defined custom function trial and error will tell me the optimal loss function
    #loss_function = functions.custom_loss_function

    ## choose ADAM otimiser (standard) with common learning rate equal to 1e-4
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    
    # Run the training loop
    #iterate over the entire training dataset for a fixed number of epochs
    for epoch in range(0, epoch_number): # 5 epochs at maximum

    # Print epoch
        print(f'Starting epoch {epoch+1}')
 
        # Iterate over the DataLoader for training data (iterate over all the batches)
        for i, data in enumerate(trainloader, 0):
            
            # Get and prepare inputs
            inputs, targets = data
            # perform some conversions (e.g. Floating point conversion and reshaping) on the inputs and targets in the current batch
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            
            # Zero the gradients: knowledge of previous improvements (especially important in batch > 0 for every epoch) is no longer available
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs).cuda()
            
            # Compute loss
            loss = loss_function(outputs, targets)
        
            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()


    # Process is complete.
    print('Training process has finished.')
    print('number of epochs ', epoch+1 )

     # save the model 
    torch.save(mlp, filename_model)

    return mlp



