#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F



# layer of the foward foward network
class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.03)
        self.threshold = 3.0
    
    def forward(self, x):
        """Forward function that takes a set of points (matrix) as input
        """
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))
    
    def forward_onesample(self, x):
        """Same as the forward fucntion but takes a vector for x and not a matrix
        """
        x_norm = x.norm(2)
        x_direction = x / (x_norm + 1e-4)
        linear_result = torch.matmul(self.weight, x_direction) + self.bias
        output = torch.relu(linear_result)
        return output
        
    def goodness_onesample(self, x):
        """ Compute the goodness for one sample
        """
        with torch.no_grad():
            goodness = self.forward_onesample(x).pow(2).mean() - self.threshold
            forwarded_x = self.forward_onesample(x)
        return goodness, forwarded_x
    
    def goodness(self, X):
        """ Compute the goodness for multiples samples sotre in a tensor matrix
        """
        with torch.no_grad():
            goodness = self.forward(X).pow(2).mean(1) - self.threshold
            forwarded_x = self.forward(X)
        return goodness, forwarded_x
    
    def train(self, x_pos, x_neg, num_epochs):
        for i in range(num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            positive_loss = F.softplus(-g_pos + self.threshold).mean()
            negative_loss = F.softplus(g_neg - self.threshold).mean()
            loss = positive_loss + negative_loss
            self.opt.zero_grad()
            # compute the gradient make a step for only one layer
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()
    
    

# Creation of the network with multiples layers
class Net(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1])]
            
    def train(self, x_pos, x_neg, num_epochs):
        """ Train network with forward-forward one layer at a time
        Args:
            x_pos (matrix of datapoints): positive data
            x_neg (matrix of datapoints): negative data
            num_epochs (int): number of epochs
        """
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg, num_epochs)
            
    def predict(self, x, Display=False):
        """Return the goodness of a given input
        Args:
            x (matrix float): Input data, can be either a vector (single sample) or a matrix (multiple samples)
            Display (bool, optional): To print stuff. Defaults to True.
        Returns:
            torch tensor float : return the total goodness
        """
        g_tot = 0
        if len(x.shape) == 1: # Case of a vector
            for i, layer in enumerate(self.layers):
                g_layer, next_x = layer.goodness_onesample(x)
                x = next_x
                g_tot += g_layer
            if Display : print('goodness total for sample : ', g_tot)
                    
        elif len(x.shape)==2:
            for i, layer in enumerate(self.layers):
                g_layer, next_x = layer.goodness(x)
                x = next_x
                g_tot += g_layer
                
        return g_tot
                

# import the dataset
def dataset_import():
    # Load the CSV file into a DataFrame
    df = pd.read_csv('../SmallDataset.csv')
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    return X, Y
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


if __name__=='__main__':
    # custom dataset into np array
    data, label =  dataset_import()
    
    x_pos = one_hot_augmentation(data, label)
    x_neg = neg_data_creation(x_pos)
    # convert to torch tensor
    x_pos = torch.tensor(x_pos).float()
    x_neg = torch.tensor(x_neg).float()
    
    # Create the network
    net = Net([5, 20, 20, 20, 20])
    net.train(x_pos, x_neg)
    
    #%% Test the network
    for i in range(len(x_pos)):
        x_test = x_neg[i].clone().detach()
        x_goodness = net.predict(x_test)
        
    # %% Test the network without a loop
    x_test = x_neg
    x_goodness = net.predict(x_test)
    print('rate of neg samples correctely labelled :', (x_goodness < 0).sum().item()/len(x_goodness))
    
    x_test = x_pos
    x_goodness = net.predict(x_test)
    print('rate of pos samples correctely labelled :', (x_goodness > 0).sum().item()/len(x_goodness))
    
    
    