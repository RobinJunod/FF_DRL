#%%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


#TODO: Try both implementation for FF learning, layerby layer or all layers
#TODO : create a last layer in order to use FF algo for regression task




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
        if x.dim() == 1:
            x_direction = x / (x.norm(2) + 1e-4)
            y = self.relu(x_direction @ self.weight.T + self.bias)
        elif x.dim() == 2:
            x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
            y = self.relu(x_direction @ self.weight.T + self.bias.unsqueeze(0))
            
        return y
    
    def goodness(self, x):
        """ Compute the goodness for multiples samples sotre in a tensor matrix
        """
        with torch.no_grad():
            if x.dim() == 1:
                goodness = self.forward(x).pow(2).mean() - self.threshold
            elif x.dim() == 2: 
                goodness = self.forward(x).pow(2).mean(1) - self.threshold
                
            forwarded_x = self.forward(x)
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
class Feature_extractor(torch.nn.Module):
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
            
    def goodness(self, x, Display=False):
        """Return the goodness of a given input
        Args:
            x (matrix float): Input data, can be either a vector (single sample) or a matrix (multiple samples)
            Display (bool, optional): To print stuff. Defaults to True.
        Returns:
            torch tensor float : return the total goodness
        """
        g_tot = 0
        for _ , layer in enumerate(self.layers):
            g_layer, next_x = layer.goodness(x)
            x = next_x
            g_tot += g_layer
        return g_tot
    
    def inference(self, input):
        """Return the "y" value (value after relu and matrix mul) 
        of all the layers
        Args:
            input (torch tensor): the input data (states)
        Returns:
            torch matrix : for each datapoint return the values at each layer
        """
        x = input
        features = torch.FloatTensor()
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                x_next= layer.forward(x)
                x = x_next
                features = torch.cat((features, x_next), dim=1)
        return features


class Regression_Layer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        """For the implementation of th FF in DQL, The input are the output of 
        every layers in the FF feature extractor. The output is the Q-value
        Args:
            in_features (int): input dimension
            out_features (int): output dimension, Q-value (nb of possible actions)
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.03)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """Forward function that takes a set of points (matrix) as input
        """
        if x.dim() == 1:
            y = self.relu(x @ self.weight.T + self.bias)
        elif x.dim() == 2:
            y = self.relu(x @ self.weight.T + self.bias.unsqueeze(0))
            
        return y
    
    def train(self, feature_extractor, input, target, num_epochs=100):
        """Train the regression layer by using the feature extractor
        Args:
            feature_extractor (Net_Feature_Extraction): FF neural net for feature extraction
            input (torch matrix): a set of datapoints (for instance a batch of states)
            target (torch vector): the regression expected value (for instance the Q value)
            num_epochs (int, optional): _description_. Defaults to 100.
        """
        with torch.no_grad():
            # features is a torch vector with the data of all layers
            features = feature_extractor.inference(input)
        
        
        for _ in range(num_epochs):
            self.opt.zero_grad()
            output = self.forward(features)
            loss = self.criterion(output, target)
            loss.backward()
            self.opt.step()

    def predict(self, feature_extractor, input):
        with torch.no_grad():
            # features is a torch vector with the data of all
            features = feature_extractor.inference(input)
            return self.forward(features)        

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
    
    # Create the network to extract features
    feature_extractor = Feature_extractor([5, 10, 10])
    feature_extractor.train(x_pos, x_neg, num_epochs=100)
    
    input = torch.cat((x_pos, x_neg), dim=0)
    size_vec = len(x_pos.T[4])
    target = torch.cat((100*torch.ones(size_vec), 1*torch.ones(size_vec)), dim=0)
    #%% Create the network to extract
    size_feature = len(feature_extractor.inference(input)[0])
    regression_layer = Regression_Layer(size_feature,1)
    regression_layer.train(feature_extractor, input, target)
    
    regression_layer.predict(feature_extractor, x_pos)
    
    # %% Test the network without a loop
    x_test = x_neg
    x_goodness = feature_extractor.goodness(x_test)
    print('rate of neg samples correctely labelled :', (x_goodness < 0).sum().item()/len(x_goodness))
    
    x_test = x_pos
    x_goodness = feature_extractor.goodness(x_test)
    print('rate of pos samples correctely labelled :', (x_goodness > 0).sum().item()/len(x_goodness))
    
    
    