import torch
import torch.optim as optim


from torch.optim.lr_scheduler import StepLR
from models.conv3d_repetition import Conv3D_Repetition

# Define the network
net = Conv3D_Repetition(num_classes=10)

optimizer = optim.RMSprop(params=net.parameters(), lr=0.01)

# Setup learning rate decay
learning_rate_scheduler = StepLR(optimizer=optimizer, step_size=100, gamma=0.1)

for i in range(1000):

    learning_rate_scheduler.step(epoch=i)
    print(optimizer.state_dict()['param_groups'][0]['lr'])

    #print(i, learning_rate_scheduler.get_lr(), optimizer.defaults)



