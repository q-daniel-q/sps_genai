"""CNN architecture for Assignment 2, adapted from SimpleCNN in Module 4 Practical 3."""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

IMAGE_SIZE = 64  # the assignment specifies a 64x64x3 input

# CIFAR10 images are 32x32, so they are resized to 64x64 to match the architecture.
# The same transform is used for training and for the API, so inputs always match.
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)


class AssignmentCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(AssignmentCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        )  # 64x64x3 -> 64x64x16
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Pooling layer, halves height and width
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        )  # 32x32x16 -> 32x32x32
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # Fully connected layer, 100 units
        self.fc2 = nn.Linear(100, num_classes)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 32x32x16
        x = self.pool(F.relu(self.conv2(x)))  # -> 16x16x32
        x = x.view(-1, 32 * 16 * 16)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x