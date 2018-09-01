import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np


# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# provide your own paths
# specification of some local drive folders to use to store the MNIST dataset (PyTorch will download the dataset into this folder for you automatically)
DATA_PATH = '/home/godfreys/Documents/Development/pytorch/tuts/basics/mnist'

# location for the trained model parameters once training is complete
MODEL_STORE_PATH = '/home/godfreys/Documents/Development/pytorch/tuts/basics/pytorch_models//'

# The mean and std deviation of the MNIST dataset
mean = 0.1307
std = 0.3081

# Componse function allows the developer to setup various manipulations on the specified dataset.
# parameters @[] array: Numerous transforms can be chained together in a list
#  first transform: converts the input data set to a PyTorch tensor
#  second transform: normalization transformation
# Neural networks train better when the input data is normalized so that the data ranges from -1 to 1 or 0 to 1
# To do this via the PyTorch Normalize transform, we need to supply the mean and standard deviation of the MNIST dataset (declared above)
# for each input channel a mean and standard deviation must be supplied (MNIST Dataset is single channeled sinc grey scale images)
# CIFAR data set, on the other hand, has 3 channels (one for each color in the RGB spectrum) you would need to provide a mean and standard deviation for each channel.

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(
    root=DATA_PATH, train=False, transform=trans)


# A data loader can be used as an iterator  
# so to extract the data we can just use the standard Python iterators such as enumerate.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Defining a model


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # nn.Sequential object:  allows us to create sequentially ordered layers in our network
        # convolution + ReLU + pooling sequence
        
        #  The argument for padding in Conv2d is 2
        filters_layer1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)

        # kernel_size is the filter size: If you wanted filters with different sized shapes in the x and y directions, you’d supply a tuple (x-size, y-size).
        # we want to down-sample our data by reducing the effective image size by a factor of 2. so set stride to 2
        pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = nn.Sequential(filters_layer1, nn.ReLU(), pooling)

        filters_layer2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.layer2 = nn.Sequential(filters_layer2, nn.ReLU(), pooling)

        # drop-out layer to avoid over-fitting 
        self.drop_out = nn.Dropout()
        
       # Fully connected layers. Just as in the case of traditional neural networks
       # nn.Linear method used to created fully connected layers in Pytorch
       # 7 * 7 * 64 = 3164
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    # It is important to call this function “forward” as this will override the base forward function in nn.Module
    # The parameter x is the data to be passed through the model. .i.e.  a batch of data
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        # Flattens the data dimensions from 7 x 7 x 64 dimensions into 3164 x 1.
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet()

# Loss and optimiser

# Loss operation that will be used to calculate the loss
# CrossEntropyLoss function combines both a SoftMax activation and a cross entropy loss function in the same function 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Run the forward pass
        # calling the forward method with the first batch of 100 imagesss
        # outputs: tensor of size (batch_size, 10)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()  # calculating the gradients 
        optimizer.step()  # and applying the gradients 

        # Track the accuracy
        total = labels.size(0)

        # max returns the index of the maximum value in a tensor.
        # @param1: The tensor to be examined 
        # @param2: The axis over which to determine the index of the maximum. 1 in this case which corresponds to 10 in the tensor (batch_size, 10)
        # To determine the model prediction, for each sample in the batch we need to find the maximum value over the 10 output nodes
        # predicted: list of prediction integers from the model 
        _, predicted = torch.max(outputs.data, 1)


        correct = (predicted == labels).sum().item()

        # number of correct items / total 
        acc_list.append(correct / total) 

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


# Test the model

# disables any drop-out or batch normalization layers in your model,
model.eval()
# disabling autograd not needed in model testing / evaluation
# and this will act to speed up the computation
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')
