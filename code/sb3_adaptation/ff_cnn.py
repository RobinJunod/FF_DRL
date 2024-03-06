#%%
from tqdm import tqdm
import torch as th
import torch.nn as nn 
from torch.optim import Adam
from sklearn.preprocessing import OneHotEncoder

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import convolve2d
from torch.functional import F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

# Mask creation for negative data
def mask_gen(image_shape=(1,28,28)):
    
    random_iter = np.random.randint(5,10)
    random_image = np.random.randint(2, size=image_shape).squeeze().astype(np.float32)
    blur_filter = np.array([[1, 2, 1], 
                            [2, 4, 2], 
                            [1, 2, 1]]) / 16
    for i in range(random_iter):
        random_image = convolve2d(random_image, blur_filter, mode='same', boundary='symm')
    mask = (random_image > 0.5).astype(np.float32)
    return mask

# Negative data generation Goeffrey Hinton
def negative_data_gen(batch):
    indexes = th.randperm(batch.shape[0])
    x1 = batch
    x2 = batch[indexes]
    mask = mask_gen()
    merged_x1 = x1*mask
    merged_x2 = x2*(1-mask)
    hybrid_image = merged_x1+merged_x2
    return hybrid_image

def show_image(x):
    x = x.squeeze()
    plt.imshow(x, cmap="gray")
    plt.show()


class FFConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FFConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.optimizer = Adam(self.parameters(), lr=0.03)
        self.threshold = 1.0
   
        
    def forward(self, x):
        # x must have shape (B x C x H x W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if not (x.dim() == 4):
            raise ValueError("Input tensor must be 4-dimensional (B x C x H x W)")
        # Frobenius norm (matrix norm) across over H x W (Normalization as in Paper Hinton)
        frobenius_norm = th.norm(x, p='fro', dim=(2, 3), keepdim=True)
        # Normalize the input tensor by dividing each element by the Frobenius norm
        x_normalized = x / (frobenius_norm + 1e-5)
        output = F.conv2d(x_normalized, self.weight, bias=self.bias, stride=self.stride, padding=self.padding) 
        return F.relu(output)
    
    def forward_ng(self, x): # Forward pass without torch grad
        with th.no_grad():
            return self.forward(x.clone())

    def goodness(self, x):
        # X is either a batch or a sample, with shape BSizex1x28x28 or 1x28x28
        output = self.forward_ng(x)
        if not (output.dim() == 3 or output.dim()==4):
            raise ValueError("Conv tensors must be 4 or 3 dim")
        if output.dim() == 3:
            output = output.unsqueeze(0)
        output = output.view(output.size(0), -1)
        # reshape ConvLay output for goodness computation
        goodness = output.pow(2).mean(1) - self.threshold
        return goodness    
    
    
    def train_ff(self, x_pos, x_neg):
        """Train the layer one step for one batch
        Args:
            x_pos (tensor): positive data
            x_neg (tensor): negative data
        Returns:
            Tuple : postive and negative data after passing trough the layer
        """
        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
        # original loss fucntion
        positive_loss = F.softplus(-g_pos + self.threshold).mean()
        negative_loss = F.softplus(g_neg - self.threshold).mean()
        loss = positive_loss + negative_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self.forward(x_pos).detach(), self.forward(x_neg).detach(), loss.item()
    

class ForwardForwardCNN(nn.Module):
    def __init__(self, n_input_channels : int = 4):
        super(ForwardForwardCNN, self).__init__()
        # Network for Breakout
        self.conv1 = FFConv2d(in_channels=n_input_channels, out_channels=64, kernel_size=10, stride=6)
        self.conv2 = FFConv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3 = FFConv2d(in_channels=128, out_channels=256, kernel_size=2)
        self.layers = [self.conv1, self.conv2, self.conv3]# get dimension of mlp output
        self._dim_output = 51_904
        
    def forward(self, obs : th.tensor):
        with th.no_grad():
            if obs.dim() == 4:
                # Debug 
                #print('obs shape', {observation.shape})
                
                #layers_output = torch.Tensor([])
                layer1 = self.conv1(obs)
                layer2 = self.conv2(layer1)
                layer3 = self.conv3(layer2)
                
                # Flatten the output of each convolutional layer
                layer1_flat = layer1.view(obs.size(0), -1)
                layer2_flat = layer2.view(obs.size(0), -1)
                layer3_flat = layer3.view(obs.size(0), -1)
                # Concatenate the flattened layers along the second dimension
                concatenated_layers = th.cat((layer1_flat, layer2_flat, layer3_flat), dim=1)
                #print('ff output shape', {concatenated_layers.shape})
            else:
                raise ValueError('Input to the netwrok must be 4 dimension : (B x C x H x W)')
            return concatenated_layers

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
    


#%% Test the FF Conv method on the mnist dataset
if __name__=='__main__':
    pass
    