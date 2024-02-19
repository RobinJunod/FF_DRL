#%%
from tqdm import tqdm
import torch 
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
def mask_gen():
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
    indexes = torch.randperm(batch.shape[0])
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
        self.epoch = 20
        
    def forward(self, x):
        # x must have shape (B x C x H x W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if not (x.dim() == 4):
            raise ValueError("Input tensor must be 4-dimensional (B x C x H x W)")
        # Frobenius norm (matrix norm) across over H x W (Normalization as in Paper Hinton)
        frobenius_norm = torch.norm(x, p='fro', dim=(2, 3), keepdim=True)
        # Normalize the input tensor by dividing each element by the Frobenius norm
        x_normalized = x / (frobenius_norm + 1e-5)
        output = F.conv2d(x_normalized, self.weight, bias=self.bias, stride=self.stride, padding=self.padding) 
        return F.relu(output)
    
    def forward_ng(self, x): # Forward pass without torch grad
        with torch.no_grad():
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
    
    def train(self, x_pos, x_neg):
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
    
    def train_layer(self, x_pos, x_neg, nb_epoch):
        """Train a layer for a number n of epochs
        Args:
            x_pos (): positive data
            x_neg (): negative data
            nb_epoch (): unmber of epoch to train single layer
        Returns:
            tuple: layer pos neg forwarded
        """
        #TODO: add validation to stop training / part in process
        for i in range(nb_epoch):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # original loss fucntion
            # loss = torch.log(1+ torch.exp(torch.cat([threshold-out_pos,out_neg-threshold]))).mean()
            positive_loss = F.softplus(-g_pos + self.threshold).mean()
            negative_loss = F.softplus(g_neg - self.threshold).mean()
            loss = positive_loss + negative_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % 10==0:
                print(f'Loss {loss}')
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()
    


class FFConvNet(nn.Module):
    """CNN multi conv layer trained with FF
    Args:
        nn (torch module): see torch doc
    """
    def __init__(self):
        super().__init__()
        
        # Network for Breakout
        self.conv1 = FFConv2d(in_channels=4, out_channels=128, kernel_size=10, stride=6)
        self.conv2 = FFConv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv3 = FFConv2d(in_channels=256, out_channels=512, kernel_size=2)
        self.layers = [self.conv1, self.conv2, self.conv3]
        
    def forward(self,x):
        with torch.no_grad():
            if x.dim() == 4:    
                #layers_output = torch.Tensor([])
                layer1 = self.conv1(x)
                layer2 = self.conv2(layer1)
                layer3 = self.conv3(layer2)
                
                # Flatten the output of each convolutional layer
                layer1_flat = layer1.view(x.size(0), -1)
                layer2_flat = layer2.view(x.size(0), -1)
                layer3_flat = layer3.view(x.size(0), -1)
                # Concatenate the flattened layers along the second dimension
                concatenated_layers = torch.cat((layer1_flat, layer2_flat, layer3_flat), dim=1)
            else:
                raise ValueError('Input to the netwrok must be 4 dimension : (B x C x H x W)')
            return concatenated_layers

    
    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        total_loss = 0
        for i, layer in enumerate(self.layers):
            h_pos, h_neg, loss_ = layer.train(h_pos, h_neg)
            total_loss += loss_
        return total_loss
    
    def train_n(self, train_loader, nb_epoch):
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train_n(train_loader, nb_epoch)
                
    def goodness(self, data):
        g_tot = 0 
        for i, layer in enumerate(self.layers):
            g = layer.goodness(data)
            g_tot += g
        return g_tot

class FeatureExtractorForwardForward(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        # Network for Breakout
        self.conv1 = FFConv2d(in_channels=n_input_channels, out_channels=128, kernel_size=10, stride=6)
        self.conv2 = FFConv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv3 = FFConv2d(in_channels=256, out_channels=512, kernel_size=2)
        self.layers = [self.conv1, self.conv2, self.conv3]
        
    def forward(self,x):
        with torch.no_grad():
            if x.dim() == 4:  
                #layers_output = torch.Tensor([])
                layer1 = self.conv1(x)
                layer2 = self.conv2(layer1)
                layer3 = self.conv3(layer2)
                
                # Flatten the output of each convolutional layer
                layer1_flat = layer1.view(x.size(0), -1)
                layer2_flat = layer2.view(x.size(0), -1)
                layer3_flat = layer3.view(x.size(0), -1)
                # Concatenate the flattened layers along the second dimension
                concatenated_layers = torch.cat((layer1_flat, layer2_flat, layer3_flat), dim=1)
            else:
                raise ValueError('Input to the netwrok must be 4 dimension : (B x C x H x W)')
            return concatenated_layers

    
    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        total_loss = 0
        for i, layer in enumerate(self.layers):
            h_pos, h_neg, loss_ = layer.train(h_pos, h_neg)
            total_loss += loss_
        return total_loss
    
    def train_n(self, train_loader, nb_epoch):
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train_n(train_loader, nb_epoch)
                
    def goodness(self, data):
        g_tot = 0 
        for i, layer in enumerate(self.layers):
            g = layer.goodness(data)
            g_tot += g
        return g_tot


class LinearClassification(nn.Module):
    """Last layer linear classifier for MNIST dataset
    Args:
        nn (torch module): see torch doc
    """
    def __init__(self, feature_extractor, input_dimension):
        super().__init__()
        self.epoch_loss = []
        self.epoch_acc = []
        self.feature_extractor = feature_extractor
        self.linear = torch.nn.Linear(input_dimension, 10)
        self.optimizer = Adam(self.parameters(), lr=learning_rate_lc)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self,x):
        return self.linear(x)
    
    def predict(self,x):
        h_activity = self.feature_extractor.respresentation_vects(x)
        output = self.forward(h_activity)
        return output.argmax(1)
    
    def accuracy_f(self, y_pred, y_true):
        # Compute the accuracy
        batch_size = y_pred.size(0)
        _, y_pred_value = y_pred.max(1)
        _, y_true_value = y_true.max(1)
        correct = torch.eq(y_pred_value, y_true_value).sum(dim=0).item()
        accuracy = correct / batch_size
        return accuracy
        
    def train(self, data_loader,epochs=20):
        evol_acc = []
        evol_loss = []
        ohe = OneHotEncoder().fit(np.arange(10).reshape((10,1))) 
        for epoch in tqdm(range(epochs), desc="Training Linear Classifier"):
            batch_loss = []
            batch_accuracy = []
            for batch in iter(data_loader):
                x,y = batch
                # OH encoding of the label
                target = torch.Tensor(ohe.transform(y.numpy().reshape(-1, 1)).toarray())
                h_activity = self.feature_extractor.respresentation_vects(x)
                output = self.forward(h_activity)
                loss = self.criterion(output,target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Logs data
                batch_loss.append(loss)
                accuracy = self.accuracy_f(output,target)
                batch_accuracy.append(float(accuracy))
            self.epoch_acc.append(float(sum(batch_accuracy)/len(batch_accuracy)))
            self.epoch_loss.append(float(sum(batch_loss)/len(batch_loss)))
            evol_acc.append(batch_accuracy)
            evol_loss.append(batch_loss)
        return evol_acc, evol_loss
    
    def test(self, data_loader):
        batch_loss = []
        batch_accuracy = []
        test_loss = 0
        ohe = OneHotEncoder().fit(np.arange(10).reshape((10,1))) 
        for batch in iter(data_loader):
            x,y = batch
            target = torch.Tensor(ohe.transform(y.numpy().reshape(-1, 1)).toarray())
            h_activity = self.feature_extractor.respresentation_vects(x)
            output = self.forward(h_activity)
            # Logs data
            loss = self.criterion(output,target)
            accuracy = self.accuracy_f(output,target)
            batch_loss.append(loss)
            batch_accuracy.append(float(accuracy))
        test_loss = float(sum(batch_loss)/len(batch_loss))
        test_accuracy = float(sum(batch_accuracy)/len(batch_accuracy))
        return test_loss,test_accuracy

#%% Test the FF Conv method on the mnist dataset
if __name__=='__main__':
    pass
    