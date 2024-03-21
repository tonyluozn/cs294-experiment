import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import math

# Data preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Model definition
class CustomCNN(nn.Module):
    def __init__(self, conv_layers, linear_layers, dropout_rate=0.5):
        super(CustomCNN, self).__init__()
        self.layers = nn.ModuleList()

        # Add convolutional layers
        for in_channels, out_channels, kernel_size, stride, padding in conv_layers:
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Flatten())

        # Add linear layers
        for in_features, out_features in linear_layers[:-1]:
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(dropout_rate))

        in_features, out_features = linear_layers[-1]
        self.layers.append(nn.Linear(in_features, out_features))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Hyperparameters and model configuration

# Conv_layer: in_channels, out_channels, kernel_size, stride, padding
# Linear_layer: in_features, out_features
    
# Model 1
conv_layers_1 = [(1, 32, 3, 1, 1),(32, 64, 3, 1, 1)]
linear_layers_1 = [(7*7*64, 128),(128, 10)]

# Model 2
conv_layers_2 = [(1, 16, 3, 1, 1),(16, 32, 3, 1, 1)]
linear_layers_2 = [(7*7*32, 128),(128, 10)]

# Model 3
conv_layers_3 = [(1, 4, 3, 1, 1),(4, 8, 3, 1, 1)]
linear_layers_3 = [(7*7*8, 128),(128, 10)]

# Model 4
conv_layers_4 = [(1, 2, 3, 1, 1),(2, 4, 3, 1, 1)]
linear_layers_4 = [(7*7*4, 128),(128, 10)]

# Model 5
conv_layers_5 = [(1, 2, 3, 1, 1),(2, 1, 3, 1, 1)]
linear_layers_5 = [(7*7*1, 128),(128, 10)]

model_hyperparameters = [(conv_layers_1, linear_layers_1), (conv_layers_2, linear_layers_2), (conv_layers_3, linear_layers_3), (conv_layers_4, linear_layers_4)]


# Compute MEC of the Decision Layer and arriving information in bits at the decision layer
def compute_information_from_train_table(train_dataset):
    digit_counts = [0] * 10
    for _, target in train_dataset:
        digit_counts[target] += 1
    total_samples = len(train_dataset)
    digit_probabilities = [count / total_samples for count in digit_counts]
    uncertainty = total_samples * sum(p * math.log2(p) for p in digit_probabilities)
    print(f"total_samples: {total_samples}")
    return - int(uncertainty)

def compute_MEC_of_decision_layer(linear_layer):
    input_size = linear_layer[0][0]
    hidden_size = linear_layer[0][1]
    return (input_size+1) * hidden_size + hidden_size

def compute_G_total(linear_layer):
    flatten_size = linear_layer[0][0]
    return (28*28*1)/flatten_size


for i, model_config in enumerate(model_hyperparameters):
    MEC_decision_layer = compute_MEC_of_decision_layer(model_config[0])
    G_total = compute_G_total(model_config[1])
    total_information = compute_information_from_train_table(train_dataset)
    bits_information_arriving_at_decision_layer = total_information/G_total if G_total >= 1 else total_information
    print(f"Model {i+1}")
    print(f"MEC of the Decision Layer: {MEC_decision_layer}")
    print(f"total_information from train table: {total_information}")
    print(f"G_total: {G_total}")
    print(f"bits_information_arriving_at_decision_layer: {bits_information_arriving_at_decision_layer}")


# Train and evaluate the model

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')

all_losses = []
for i, model_config in enumerate(model_hyperparameters):
    print(f"Model {i+1}")
    model = CustomCNN(model_config[0], model_config[1], dropout_rate=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = train_model(model, train_loader, criterion, optimizer, num_epochs=10)
    all_losses.append(loss)
    evaluate_model(model, test_loader)


num_epochs = 10
# Plotting the losses
plt.figure(figsize=(10, 6)) 
for i, losses in enumerate(all_losses):
    epochs = list(range(1, num_epochs + 1))
    plt.plot(epochs, losses, label=f'Model {i+1}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss of Different Models')
plt.legend()
plt.grid(True)
plt.show()