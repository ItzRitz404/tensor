# Building the Neural Network

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# get a hardware device for training

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# define the class


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

'''We create an instance of NeuralNetwork, move it to the device, and print its structure.'''

model = NeuralNetwork().to(device)
print(model)

# We get the prediction densities by passing it through an instance of the nn.Softmax.

X = torch.rand(1, 28, 28, device=device)
logits = model(X) 
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# weigths and biases

print(f"First Linear weights: {model.linear_relu_stack[0].weight} \n")

print(f"First Linear biases: {model.linear_relu_stack[0].bias} \n")

#  model layers

input_image = torch.rand(3,28,28)
print(input_image.size())

# nn.Flatten

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())