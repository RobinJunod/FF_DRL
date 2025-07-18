#%%
import argparse
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from Dataset import dataset_GMM, fake_data_shuffle


# layer of the foward foward network
class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.03)
        # TODO: in original paper treshold = nb of neurons, 1:1
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
            
    def train_L(self, x_pos, x_neg, num_epochs):
        """ Train network with forward-forward one layer at a time
        Args:
            x_pos (matrix of datapoints): positive data
            x_neg (matrix of datapoints): negative data
            num_epochs (int): number of epochs
        """
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            # print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg, num_epochs)
    
    def train_B(self, x_pos, x_neg, num_epochs):
        """ Train network with forward-forward one batch at a time.
        Pass through the entier network num_epochs times.
        Args:
            x_pos (matrix of datapoints): positive data
            x_neg (matrix of datapoints): negative data
            num_epochs (int): number of epochs
        """
        for epoch in range(num_epochs):
            h_pos, h_neg = x_pos, x_neg
            print(f'training epoch : {epoch}')
            for i, layer in enumerate(self.layers):
                h_pos, h_neg = layer.train(h_pos, h_neg, 1)
            
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
    
    def inference(self, input, n_layer=0):
        """Return the "y" value (value after relu and matrix mul) 
        of all the layers
        Args:
            input (torch tensor): the input data (states)
            start_layer (int) : the layer number from which we want to 
                                take features, all the firsts layer are ignored
        Returns:
            torch matrix : for each datapoint return the values at each layer
        """
        x = input
        features = torch.FloatTensor()
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                x_next= layer.forward(x)
                x = x_next
                if i >= n_layer:
                    if x.dim() == 1:
                        # Inference for 1 data (1 state)
                        features = torch.cat((features, x_next), dim=0)
                    elif x.dim() == 2:
                        # Inference for multiples data (list of states)
                        features = torch.cat((features, x_next), dim=1)
        return features

# Copy of the FF network for performence comparaison with backprop
class Backprop_net(torch.nn.Module):
    def __init__(self, dims):
        super(Backprop_net, self).__init__()
        self.layers = nn.ModuleList()  # Use ModuleList to hold the layers
        self.LLDim = dims[len(dims) - 1]
        for d in range(len(dims) - 1):
            # Define linear layer and ReLU activation separately
            linear_layer = nn.Linear(dims[d], dims[d + 1])
            relu = nn.ReLU()
            # Add to the list of layers
            self.layers.extend([linear_layer, relu])
        # Regression layer (output layer)
        self.regression_layer = nn.Linear(self.LLDim, 1)

    def forward(self, x):
        # Implement the forward pass by iterating through the layers
        for layer in self.layers:
            x = layer(x)
        # Apply regression layer to the output of the last hidden layer
        x = self.regression_layer(x)
        return x


#%%
if __name__=='__main__':
    
    # __________PARSERS__________
    parser = argparse.ArgumentParser(description="A simple command-line parser.")
    # Add command-line arguments
    parser.add_argument("--train_ff", type=bool, default=True, help="if want to train the mdoel with FF")
    parser.add_argument("--train_bp", type=bool, default=False, help="if want to train the mdoel with bp")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    # Parse the command-line arguments
    args = parser.parse_args()
    # CREATION OF THE DATASET
    X_train, X_test = dataset_GMM(n_samples= 10000, show_plot=False)
    # create dataframe
    df_train =  pd.DataFrame(X_train, columns=['dim1', 'dim2', 'dim3', 'dim4'])
    df_test =  pd.DataFrame(X_test, columns=['dim1', 'dim2', 'dim3', 'dim4'])
    # 3 - Define a target as a nonlinear combination of the inputs 
    # the formula is : 2*x1 + x2^2 +x3*x4
    df_train['target_nonlinear'] = 2 * df_train['dim1'] + df_train['dim2']**2 + \
                                       df_train['dim3'] * df_train['dim4']
    df_test['target_nonlinear'] = 2 * df_test['dim1'] + df_test['dim2']**2 + \
                                      df_test['dim3'] * df_test['dim4']
    Y_train = torch.tensor(df_train['target_nonlinear']).float()
    Y_test = torch.tensor(df_test['target_nonlinear']).float()
    # 4 - Train FF with dimensionality shuffling with a simulated dataset with 4 inputs
    positive_data = torch.tensor(X_train).float()
    positive_data, negative_data = fake_data_shuffle(positive_data)
    # Dimension of the FF networks layers
    dims = [4, 10, 5]
    
    num_epochs = args.num_epochs
    #%% Train with FF algorithm
    if args.train_ff:
        R2_evolution = []
        df = pd.DataFrame()
        for n_train in range(20): # train with 20 diff FeatureExctractor
            for epoch in range(1,num_epochs, 1):
                # Create/train the FF net to extract features
                feature_extractor = Feature_extractor(dims)
                feature_extractor.train_L(positive_data, negative_data, num_epochs=epoch)
                # Create the last layer for regression and train it
                size_feature = len(feature_extractor.inference(positive_data)[0])
                regression_layer = nn.Linear(size_feature,1)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(regression_layer.parameters(), lr=0.01)
                for n in range(epoch):
                    # Forward pass
                    # print('pos data :', positive_data.shape)
                    features = feature_extractor.inference(positive_data).float()
                    # print('features :', features.shape)
                    outputs = regression_layer(features).float()  # Reshape x to a 2D tensor
                    # Compute the loss
                    loss = criterion(outputs, Y_train.view(-1, 1))  # Reshape y to a 2D tensor
                    # Backpropagation and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # Test the model on the test dataset
                test_data = torch.tensor(X_test).float()
                pos_test_data, neg_test_data = fake_data_shuffle(test_data)
                test_features = feature_extractor.inference(pos_test_data).float()
                test_outputs = regression_layer(test_features).float()
                # R2 metric for this regression problem
                y_true = Y_test.numpy()
                y_pred = test_outputs.detach().numpy()
                r2 = r2_score(y_true, y_pred)
                print(f'FE {n_train}, epoch{epoch}, R2 score : {r2}')
                R2_evolution.append((epoch,r2))
            
            if n_train==1:
                # Save evolution R2 score
                df = pd.DataFrame(R2_evolution, columns=['epoch', 'r2score'])
            else:
                _, r2 = zip(*R2_evolution)
                df[f'r2score{n_train}'] = r2
            
            R2_evolution.clear()
        
        # Save the DataFrames to a CSV file
        df.to_csv(f'../results/FF_Qlearning/logs/r2_regresProblem_{epoch}_20.csv', index=False)
        

    #%% Train  with BackPropagation
    if args.train_bp:
        R2_evolution_bp = []
        df1 = pd.DataFrame()
        for n_train in range(20):  
            backprop_network = Backprop_net(dims)
            criterion_bp = nn.MSELoss()
            optimizer_bp = torch.optim.Adam(backprop_network.parameters(), lr=0.01)
            for epoch in range(num_epochs):
                # Forward pass
                outputs = backprop_network(positive_data).float()  # Reshape x to a 2D tensor
                # Compute the loss
                loss = criterion_bp(outputs, Y_train.view(-1, 1))  # Reshape y to a 2D tensor
                # Backpropagation and optimization
                optimizer_bp.zero_grad()
                loss.backward()
                optimizer_bp.step()
                # Print the loss
                #if (epoch + 1) % 100 == 0:
                #    print(f'Last layer , Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

                # Test the model on the test dataset
                test_data = torch.tensor(X_test).float()
                test_outputs = backprop_network(test_data).float()
                # R2 metric for this regression problem
                y_true = Y_test.numpy()
                y_pred = test_outputs.detach().numpy()
                r2 = r2_score(y_true, y_pred)
                print('R2 score : ', r2)
                R2_evolution_bp.append((epoch, r2))
                
            if n_train==1:
                # Save evolution R2 score
                df1 = pd.DataFrame(R2_evolution_bp, columns=['epoch', 'r2score'])
                print(df1)
            else:
                _, r2 = zip(*R2_evolution_bp)
                df1[f'r2score{n_train}'] = r2
            R2_evolution_bp.clear()
        # Save the DataFrames to a CSV file
        df1.to_csv(f'../results/FF_Qlearning/logs/r2_regresProblem_bp_{epoch}_20.csv', index=False)
        
    
    #%% Train with linear model
    #regression_layer = nn.Linear(4,1)
    #criterion = nn.MSELoss()
    #optimizer = torch.optim.Adam(regression_layer.parameters(), lr=0.01)
    ## Train the model using the FF algorithm
    #num_epochs = 300
    #for epoch in range(num_epochs):
    #    # Forward pass
    #    outputs = regression_layer(positive_data).float()
    #    # Compute the loss
    #    loss = criterion(outputs, Y_train.view(-1, 1))  # Reshape y to a 2D tensor
    #    # Backpropagation and optimization
    #    optimizer.zero_grad()
    #    loss.backward()
    #    optimizer.step()
    #    # Print the loss
    #    if (epoch + 1) % 100 == 0:
    #        print(f'Last layer , Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    ## Test the model on the test dataset
    #test_data = torch.tensor(X_test).float()
    #pos_test_data, neg_test_data = fake_data_shuffle(test_data)
    #test_outputs = regression_layer(test_data).float()
    ## R2 metric for this regression problem
    #y_true = Y_test.numpy()
    #y_pred = test_outputs.detach().numpy()
    #r2 = r2_score(y_true, y_pred)
    #print('R2 score : ', r2)
    