import config
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error 
import hydroeval as he

class MLP_model(torch.nn.Module):

## init the superclass
  def __init__(self):
    super().__init__()
    '''
    input first flows through the first layer, followed by the second, followed by..
    '''
    self.layers = torch.nn.Sequential(
      torch.nn.Linear(config.l0, config.l1),
      torch.nn.ReLU(),
      torch.nn.Linear(config.l1, config.l2),
      torch.nn.ReLU(),
      torch.nn.Linear(config.l2, 1)
    ).cuda()

    
  def forward(self, x):
    '''
      Forward pass: feed the input data through the model (self.layers) and return the result
    '''
    return self.layers(x)


'''
Optimizing notes:
- Using torch.Tensor instead of torch.from_numpy(): Instead of converting the numpy arrays to PyTorch tensors using 
torch.from_numpy(), we can directly create PyTorch tensors from the numpy arrays using torch.Tensor(). This is 
because torch.Tensor() automatically uses the same data type and device as the input array, which saves us from 
having to call .cuda() later on.
- Caching the scaled data: If the data needs to be scaled, it would be more efficient to cache the scaled data 
instead of scaling it every time the dataset is initialized. This way, if the dataset is used multiple times, the 
scaling is only done once.
'''
class BarraDataset(torch.utils.data.Dataset):
    '''
    Convert the data to the right format using a Torch Tensor.
    '''
    def __init__(self, X, y, scale_data=False):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

        if scale_data:
            scaler = StandardScaler()
            self.X = torch.Tensor(scaler.fit_transform(self.X.numpy()))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        return x, y

'''
Optimizing notes:
- Usually good to check the input data for missing values and handle them appropriately before computing the performance 
metrics. Currently, the function assumes that the input data is complete and does not check for missing values.
- Instead of computing the descriptive statistics for the target and predicted values separately, we can compute them together 
using the describe() function of the Pandas DataFrame. This can make the code more concise and easier to read.
'''
def testing_performance(test_df, current_model, year):
    '''
    Calculate the metric of performance prediction against target
    '''
    try:
        test_df.dropna(inplace=True)
    except:
        raise ValueError('Input data contains missing values')
    
    kge, r, alpha, beta = he.evaluator(he.kge, test_df['target'].to_numpy(), test_df[current_model].to_numpy())
    corr = test_df[['target', current_model]].corr().iloc[0, 1]
    rmse = mean_squared_error(test_df['target'], test_df[current_model], squared=False)
    describe_df = test_df[['target', current_model]].describe()

    columns = ['year', 'kge', 'r', 'alpha', 'beta', 'cor', 'rmse', 'mean_target', 'mean_pred',
               'std_target', 'std_pred', 'min_target', 'min_pred', '25%_target', '25%_pred',
               '50%_target', '50%_pred', '75%_target', '75%_pred', 'max_target', 'max_pred']
    data = [[year, kge, r, alpha, beta, corr, rmse] + describe_df.loc['mean':, col].tolist()
            for col in ['target', current_model]]

    eval_data = pd.DataFrame(data, columns=columns)

    return eval_data

def train_mlp(mlp, trainloader, optimizer, loss_function, seed, epoch_number, batch_size):
    '''
    Function that trains a model based on the input data, etc.
    '''
  
    # Set fixed random number seed
    torch.manual_seed(seed)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp.to(device)  # Move the MLP model to GPU if available
    
    # Run the training loop
    for epoch in range(epoch_number):
        print(f'Starting epoch {epoch+1}')
 
        # Iterate over the DataLoader for training data.
        for i, data in enumerate(trainloader, 0):
            # Get and prepare inputs
            inputs, targets = data
            # perform some conversions (e.g. Floating point conversion and reshaping) on the inputs and targets in the current batch
            inputs, targets = inputs.float().to(device), targets.float().to(device)  # Move inputs and targets to GPU if available
            targets = targets.reshape((targets.shape[0], 1))
            
            # Zero the gradients: knowledge of previous improvements (especially important in batch > 0 for every epoch) is no longer available
            optimizer.zero_grad()
            # Perform forward pass
            outputs = mlp(inputs)
            # Compute loss
            loss = loss_function(outputs, targets)
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()

    print('Training process has finished.')
    print('Number of epochs:', epoch+1 )

    return mlp