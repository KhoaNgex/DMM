# %%
import deepmatcher as dm
# %%
import torch
torch.cuda.is_available()
# %%
train, validation, test = dm.data.process(
    path='sample_data/itunes-amazon',
    train='train.csv',
    validation='validation.csv',
    test='test.csv')
# %%
### Peeking at processed data
train_table = train.get_raw_table()
train_table.head()
# %%
# Define neural network model
