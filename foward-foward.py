#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# import the dataset
def dataset_import():
    # Load the CSV file into a DataFrame
    df = pd.read_csv('SmallDataset.csv')
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    return X, Y

# add the label to the input data 
# input -> numpy arrays of the data and the labels
def one_hot_augmentation(x, y):
    # x for data, y for labels
    unique_values = np.unique(y)
    one_hot_matrix = np.zeros((len(y), len(unique_values)), dtype=int)

    for i, label in enumerate(y):
        one_hot_matrix[i, label] = 1
    
    x_pos = np.hstack((x, one_hot_matrix))
    return x_pos

# Create negative data 
def neg_data_creation(x_pos):
    x_neg  = x_pos[:, [0, 1, 2, 4, 3]]
    return x_neg

## Simple network build with torch
#class SimpleNN(nn.Module):
#    def __init__(self, input_size, hidden_size, output_size):
#        super(SimpleNN, self).__init__()
#        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
#        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer
#    def forward(self, x):
#        x = F.relu(self.fc1(x))  # Apply ReLU activation to the hidden layer
#        x = self.fc2(x)  # Output layer
#        return F.softmax(x, dim=1)  # Apply softmax activation to the output


## Activation function (sigmoid) for the last layer
#def Sigmoid(x):
#    return 1 / (1 + torch.exp(-x))
## ReLu
#def ReLu(x):
#    return x if x > 0 else 0
#def ReLu_derivate(x):
#    return 1 if x > 0 else 0
#
## To use after each layers
#def Normalize(x):
#    return x / (torch.norm(x, p=2) + 1.e4)
#
## compute the loss
#def loss(g_pos, g_neg, threshold):
#    torch.log(1 + torch.exp(torch.cat([
#                -g_pos + threshold,
#                g_neg - threshold]))).mean()


# layer of the foward foward network
class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.SGD(self.parameters(), lr=0.03, momentum=0.9)
        self.threshold = 2.0
        self.num_epochs = 200
    
    # The foward pass
    # def forward(self, x):
    #     # x must be a torch tensor
    #     x_normalized = x / (torch.norm(x, p=2) + 1.e4)
    #     print(x_normalized)
    #     print(self.weight.T)
    #     x_ =  torch.mm(x_normalized, self.weight.T) + self.bias.unsqueeze(0)
    #     x_ = self.relu(x_)
    #     return x_
    
    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))
    
    def forward_inference(self, x):
        x_norm = x.norm(2)
        x_direction = x / (x_norm + 1e-4)
        linear_result = torch.matmul(self.weight, x_direction) + self.bias
        output = torch.relu(linear_result)
        return output
        
    def train(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            positive_loss = torch.log(1 + torch.exp(-g_pos + self.threshold)).mean()
            negative_loss = torch.log(1 + torch.exp(g_neg - self.threshold)).mean()
            #loss = torch.log(1 + torch.exp(torch.cat([
            #    -g_pos + self.threshold,
            #    g_neg - self.threshold]))).mean()
            loss = positive_loss + negative_loss
            self.opt.zero_grad()
            # compute the gradient make a step for only one layer
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

    def goodness(self, x):
        with torch.no_grad():
            goodness = self.forward_inference(x).pow(2).mean() - self.threshold
            forwarded_x = self.forward_inference(x)
        return goodness, forwarded_x
    
# Creation of the network with multiples layers
class Net(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1])]
            
    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)
            

    def predict(self, x, Display=False):
        g_tot = 0
        for i, layer in enumerate(self.layers):
            g_layer, next_x = layer.goodness(x)
            x = next_x
            
            # print('goodness at layer', i, ' : ', g_layer)
            g_tot += g_layer
        
        print('goodness total : ', g_tot)
        if Display:
            if g_tot > 0:
                print('the data is positive, overall goodness : ', g_tot.item())
            else:
                print('the data is negative, overall goodness : ', g_tot.item())
        
        

if __name__=='__main__':
    # custom dataset into np array
    data, label =  dataset_import()
    x_pos = one_hot_augmentation(data, label)
    x_neg = neg_data_creation(x_pos)
    # convert to torch tensor
    x_pos = torch.tensor(x_pos).float()
    x_neg = torch.tensor(x_neg).float()
    
    # Create the network
    net = Net([5, 10, 10])
    net.train(x_pos, x_neg)
    
    #%% Test the network
    for i in range(len(x_pos)):
        x_test = x_neg[i].clone().detach()
        net.predict(x_test)
    # %%
    
    