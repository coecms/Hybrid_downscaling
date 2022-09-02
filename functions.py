
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error 
import hydroeval as he

# Convert the data to the right format
class BarraDataset(torch.utils.data.Dataset):
  

  def __init__(self, X, y, scale_data=False): 
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if not applied elsewhere
      if scale_data:
          X = StandardScaler().fit_transform(X)
          print('scale')
      self.X = torch.from_numpy(X).cuda()
      self.y = torch.from_numpy(y).cuda()

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

# Calculate the metric of performance prediction against target
def testing_performance(test_df, current_model, year):
    kge, r, alpha, beta = he.evaluator(he.kge, test_df['target'].to_numpy(), test_df[current_model].to_numpy())
    correlation = test_df['target'].corr(test_df[current_model])
    rmse = mean_squared_error(test_df['target'], test_df[current_model], squared = False)
    describe_df = test_df[['target',current_model]].describe()


    columns = ['year','kge', 'r', 'alpha', 'beta', 'cor', 'rmse', 'mean_target', 'mean_pred',
    'sd_target', 'sd_pred', 'min_target', 'min_pred',
    'perc25_target', 'perc25_pred', 'perc50_target', 'perc50_pred',
    'perc75_target', 'perc75_pred', 'max_target', 'max_pred' ]
    data = [[year, float(kge), float(r), float(alpha), float(beta), correlation, rmse, describe_df.iat[1,0], describe_df.iat[1,1],
    describe_df.iat[2,0], describe_df.iat[2,1], describe_df.iat[3,0], describe_df.iat[3,1],
    describe_df.iat[4,0], describe_df.iat[4,1], describe_df.iat[5,0], describe_df.iat[5,1],
    describe_df.iat[6,0], describe_df.iat[6,1],describe_df.iat[7,0], describe_df.iat[7,1]]]

    eval_data = pd.DataFrame(data)
    eval_data.columns = columns

    return(eval_data)
    
## Manually define a loss function -- I ended up not using this
def custom_loss_function(output, target):
    square_difference = torch.square(output - target)
    mean_square_difference = torch.mean(square_difference)
    root_mean_square_difference =torch.sqrt(mean_square_difference)

    min_output = torch.min(output)
    min_target = torch.min(target)
    min_difference = torch.abs( min_output - min_target)

    max_output = torch.max(output)
    max_target = torch.max(target)
    max_difference = torch.abs( max_output - max_target)

    sd_output = torch.std(output)
    sd_target = torch.std(target)
    sd_difference = torch.abs(sd_output - sd_target)

    loss_value = 0.1 * sd_difference + 0.2 *min_difference + 0.3* max_difference +  0.4* root_mean_square_difference

    return(loss_value)

## Manually define a loss function -- I ended up not using this
def custom_loss_function2(output, target):

    mean_target = torch.mean(target)
    square_difference = torch.square(output - target)
    mean_square_difference = torch.mean(square_difference)
    root_mean_square_difference =torch.sqrt(mean_square_difference)
    relative_mean_square_difference = torch.div(root_mean_square_difference, mean_target)

    min_output = torch.min(output)
    min_target = torch.min(target)
    min_difference = torch.abs( min_output - min_target)
    relative_min_difference = torch.div(min_difference, mean_target)

    max_output = torch.max(output)
    max_target = torch.max(target)
    max_difference = torch.abs( max_output - max_target)
    relative_max_difference = torch.div(max_difference, mean_target)

    sd_output = torch.std(output)
    sd_target = torch.std(target)
    sd_difference = torch.abs( sd_output - sd_target)
    relative_sd_difference = torch.div(sd_difference, sd_target)
    loss_value = 0.2 *relative_min_difference + 0.2* relative_max_difference +  0.6 * relative_mean_square_difference

    return(loss_value)


