#%%
from tqdm import tqdm
import torch 
import torch.nn as nn 
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder


import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import convolve2d
from torch.functional import F


# Mask creation for negative data
def mask_gen():
    random_iter = np.random.randint(5,10)
    random_image = np.random.randint(2, size=image_shape).squeeze().astype(np.float32)
    blur_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
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
    
    def train_n(self, x_pos, x_neg, nb_epoch):
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
        # Network for MNIST
        # self.conv1 = FFConv2d(in_channels=1, out_channels=128, kernel_size=10, stride=6)
        # self.conv2 = FFConv2d(in_channels=128, out_channels=220, kernel_size=3)
        # self.conv3 = FFConv2d(in_channels=220, out_channels=512, kernel_size=2)
        # self.layers = [self.conv1, self.conv2, self.conv3]
        
        # Network for Breakout
        self.conv1 = FFConv2d(in_channels=4, out_channels=128, kernel_size=10, stride=6)
        self.conv2 = FFConv2d(in_channels=128, out_channels=220, kernel_size=3)
        self.conv3 = FFConv2d(in_channels=220, out_channels=512, kernel_size=2)
        self.layers = [self.conv1, self.conv2, self.conv3]
        
    def respresentation_vects(self,x):
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
                raise ValueError('Input to the netwrok must ne 4 dimension : (B x C x H x W)')
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

# Test the FF Conv method on the mnist dataset
if __name__=='__main__':
    batchsize = 1024
    learning_rate = 0.03
    learning_rate_lc = 0.05
    epochs = 5
    threshold = 1.0
    image_shape = (1,28,28)
    
    train_dataset = MNIST("./data/",download=True, train=True, transform=ToTensor())
    test_dataset = MNIST("./data/",download=True, train=False, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)
    
    FF_feature_extractor = FFConvNet()

    #%% Trainig method 1 
    """
    There are two approaches for batch training:
    1. Train batches for all layers. ---> straigth forward
    2. Train batches for each layer. ---> need to create new batches for next layer input
    We use 1 for the following two training methods.
    """
    for epoch in range(epochs):
        epoch_loss_mean = 0
        for i, data in enumerate(train_loader, 0):
            
            inputs, labels = data
            #print(f'batch training number {i} / {len(train_loader)}')
            x_pos = inputs
            x_neg = negative_data_gen(inputs)
            loss = FF_feature_extractor.train(x_pos, x_neg)
            epoch_loss_mean = (i*epoch_loss_mean + loss)/(i+1)
            
        print(f'Loss mean of the epoch {epoch_loss_mean}')
    
    #feature_extractor_output = FF_feature_extractor.respresentation_vects(next(iter(train_loader))[0]).shape[1] # With current feature extractor
    feature_extractor_output = 3440 # With current feature extractor
    #%% Training method 1 Linear classifier
    Linear_classifier = LinearClassification(FF_feature_extractor, input_dimension=feature_extractor_output)
    Linear_classifier.train(train_loader, epochs=10)
    
    test_loss,test_acc = Linear_classifier.test(test_loader)
    print("Test Loss: ",test_loss)
    print("Test Accuracy: ",test_acc)
    
    #%% Trainig method 2 
    """
    There are two approaches for batch training:
    1. Train batches for each layer. ---> need to create new batches for next layer input
    This method needs more memory capacity
    """
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        #print(f'batch training number {i} / {len(train_loader)}')
        x_pos = inputs
        x_neg = negative_data_gen(inputs)
        loss = FF_feature_extractor.train(x_pos, x_neg)
    
    epoch_loss_mean = (i*epoch_loss_mean + loss)/(i+1)
    
    print(f'Loss mean of the epoch {epoch_loss_mean}')


    #%% test model
    avg_accruacy = 0
    i = -1
    for _, data in enumerate(test_loader, 0):
        i += 1
        inputs, labels = data
        x_pos = inputs
        x_neg = negative_data_gen(inputs)
        
        avg_accruacy = (i*avg_accruacy + (FF_feature_extractor.conv1.goodness(x_pos)>0).sum()/len(FF_feature_extractor.conv1.goodness(x_pos)))/(i+1)
        
        print(f'accruacy of the pos data {(FF_feature_extractor.conv1.goodness(x_pos)>0).sum()/len(FF_feature_extractor.conv1.goodness(x_pos))}')
        print(f'accruacy of the neg data {(FF_feature_extractor.conv1.goodness(x_neg)<0).sum()/len(FF_feature_extractor.conv1.goodness(x_neg))}')
        
    print(f'Average accruacy : {avg_accruacy}')
# %%
    # Save the model after training
    model.save_model()
    # Loading the model
    loaded_model = FFConvNet()
    loaded_model.load_state_dict(torch.load('ffconvnet_model.pth'))
    loaded_model.eval() 
#%%

"""
class FFLinL(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.03)
        # TODO: in original paper treshold = nb of neurons, 1:1
        self.threshold = 1.0
    
    def forward(self, x):
        # x must have shape (B x D)
        if not (x.dim() == 1 or x.dim()==2):
            raise ValueError("Conv tensors must be 4 or 3 dim")
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        y = self.relu(x_direction @ self.weight.T + self.bias.unsqueeze(0))
        return y
 
    def forward_ng(self, x): # Forward pass without torch grad
        with torch.no_grad():
            return self.forward(x.clone())    
    
    def goodness(self, x):
        output = self.forward_ng(x)
        if not (output.dim() == 3 or output.dim()==4):
            raise ValueError("Conv tensors must be 4 or 3 dim")
        if output.dim() == 3:
            output = output.unsqueeze(0)
        output = output.view(output.size(0), -1)
        # reshape ConvLay output for goodness computation
        goodness = output.pow(2).mean(1) - self.threshold
        return goodness
    
    def train(self, x_pos, x_neg, num_epochs):
        for i in range(num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            positive_loss = F.softplus(-g_pos + self.threshold).mean()
            negative_loss = F.softplus(g_neg - self.threshold).mean()
            loss = positive_loss + negative_loss
            self.opt.zero_grad()
            # compute the gradient make a step for only one layer
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()
"""