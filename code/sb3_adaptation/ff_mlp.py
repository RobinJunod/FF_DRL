import random

import torch as th
import torch.nn as nn
import torch.nn.functional as F


## Negative data generator 
def negative_shuffling(postive_data : th.tensor):
    """Schuffle the data randomely to create fake data.
    The data dimensio are kept , only the values are changing
    Args:
        real_data_tensor (2d torch tensor): the "xpos", a 2d pytorch tensor (multiples state action pairs) 
    Returns:
        pos_data, neg_data (torch tensor): the positive and negative data in torch tensors
    """
    # Transpose the tensor
    postive_data_T_list = postive_data.t().tolist()
    # Shuffle the values in each inner list
    for inner_list in postive_data_T_list:
        random.shuffle(inner_list)
    
    negative_data = th.tensor(postive_data_T_list).T
    # Create a mask to remove the data that are similar 
    mask = th.all(negative_data == postive_data, dim=1)
    # Use the mask to remove rows from both tensors
    real_data= postive_data[~mask]
    fake_data= negative_data[~mask]
    return real_data, fake_data




# layer of the foward foward network
class FFLayer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = th.nn.ReLU()
        self.opt = th.optim.Adam(self.parameters(), lr=0.03)
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
        with th.no_grad():
            if x.dim() == 1:
                goodness = self.forward(x).pow(2).mean() - self.threshold
            elif x.dim() == 2: 
                goodness = self.forward(x).pow(2).mean(1) - self.threshold
            forwarded_x = self.forward(x)
        return goodness, forwarded_x
    
    def train_ff(self, x_pos, x_neg, num_epochs):
        for _ in range(num_epochs):
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


    
class ForwardForwardMLP(nn.Module):
    def __init__(self, dims):
        super(ForwardForwardMLP, self).__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [FFLayer(dims[d], dims[d + 1])]
        self._dim_ouput =  sum(dims[1:]) # get dimension of mlp output
        
    def forward(self, obs : th.tensor):
        """Return the "y" value (value after relu and matrix mul) 
        of all the layers
        Args:
            obs (torch tensor): the input data (states)
        Returns:
            torch matrix : for each datapoint return the values at each layer
        """
        #print('obs shape', obs.shape)
        features = th.FloatTensor()
        with th.no_grad():
            for _, layer in enumerate(self.layers):
                obs_next = layer.forward(obs)
                obs = obs_next
                if obs.dim() == 1:
                    # Inference for 1 data (1 state)
                    features = th.cat((features, obs_next), dim=0)
                elif obs.dim() == 2:
                    # Inference for multiples data (list of states)
                    features = th.cat((features, obs_next), dim=1)
        return features

    def train_ff(self, x_pos, x_neg, num_epochs):
        """ Train network with forward-forward one batch at a time.
        Pass through the entier network num_epochs times.
        Args:
            x_pos (matrix of datapoints): positive data
            x_neg (matrix of datapoints): negative data
            num_epochs (int): number of epochs
        """
        for epoch in range(num_epochs):
            h_pos, h_neg = x_pos, x_neg
            #print(f'training epoch : {epoch}')
            for i, layer in enumerate(self.layers):
                h_pos, h_neg = layer.train_ff(h_pos, h_neg, 1)
                
    def train_ll(self, x_pos, x_neg, num_epochs):
        """ Train network with forward-forward layer by layer
        Args:
            x_pos (matrix of datapoints): positive data
            x_neg (matrix of datapoints): negative data
            num_epochs (int): number of epochs
        """
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print(f'train layer {i}')
            h_pos, h_neg = layer.train_ff(h_pos, h_neg, num_epochs)
                
    def goodness(self, input, Display=False):
        """Return the goodness of a given input
        Args:
            x (matrix float): Input data, can be either a vector (single sample) or a matrix (multiple samples)
            Display (bool, optional): To print stuff. Defaults to True.
        Returns:
            torch tensor float : return the total goodness
        """
        g_tot = 0
        for _ , layer in enumerate(self.layers):
            g_layer, next_input = layer.goodness(input)
            input = next_input
            g_tot += g_layer
        return g_tot
    

