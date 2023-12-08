import torch
import torch.nn as nn

# Define the Q-network
class QNetwork(nn.Module):
    """Network to optimize the DQN with backpropagation
    Takes as input :  B x C x H x W format
    Args:
        num action (int): the number of possible actions
    """
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136 , 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten tensor
        x = torch.relu(self.fc1(x))
        return self.fc2(x)