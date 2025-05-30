import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from model import MNISTClassifier

# set numpy to seed for consistent results
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28 * 28  # Flattened image size
hidden_size = 256
num_classes = 10  # Digits 0-9
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)    

def prune_weights(model, masks, prune_threshold=0.5):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                nonzero_weights = param[param != 0].abs()
                if nonzero_weights.numel() == 0:
                    continue  # Skip if there are no nonzero weights
                threshold = torch.quantile(nonzero_weights, 1 - prune_threshold)
                mask = (param.abs() >= threshold).float()
                masks[name] = mask
                param *= mask

def enforce_pruning(model, masks):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param *= masks[name]   # keep pruned weights at zero

def count_zero_nonzero_weights(model):
    total_weights = 0
    zero_weights = 0

    for name, param in model.named_parameters():
        if "weight" in name:
            total_weights += param.numel()
            zero_weights += torch.sum(param == 0).item()

    nonzero_weights = total_weights - zero_weights
    zero_percentage = (zero_weights / total_weights) * 100
    nonzero_percentage = (nonzero_weights / total_weights) * 100

    print(f"Total Weights: {total_weights}")
    print(f"Zero Weights: {zero_weights} ({zero_percentage:.2f}%)")
    print(f"Nonzero Weights: {nonzero_weights} ({nonzero_percentage:.2f}%)")

def get_weight_distribution(model):
    weights = []
    for name, module in model.named_parameters():
        if 'weight' in name:
            weights.extend(module.detach().cpu().numpy().flatten())
    
    plt.hist(weights, bins=50, alpha=0.75, color='blue')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Weight Distribution of Model')
    plt.show()

model = MNISTClassifier(input_size, num_classes, hidden_size=hidden_size).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if os.path.exists("pruned_model.pth"):
    model.load_state_dict(torch.load("pruned_model.pth", weights_only=True))
else:
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    if not os.path.exists("unpruned_model.pth"):    
        torch.save(model.state_dict(), "unpruned_model.pth")
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Initial Test Accuracy: {100. * accuracy:.2f}%")

    fine_tune_iterations = 5

    masks = {}

    for i in range(fine_tune_iterations):
        prune_weights(model, masks, prune_threshold=0.5)

        count_zero_nonzero_weights(model)

        for epoch in range(num_epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                enforce_pruning(model, masks)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy after Prune #{i+1}: {100. * accuracy:.2f}%")

    torch.save(model.state_dict(), "pruned_model.pth")

count_zero_nonzero_weights(model)

def plot_accuracy_no_correction_vs_correction(model, tests):
    error_rates = [0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    unsupressed_accuracies = []
    suppressed_accuracies = []

    model.eval()
    for error_rate in error_rates:
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tests:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, inject_faults=True, suppress_errors=False, error_rate=error_rate)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy with Error Rate {error_rate}: {100. * accuracy:.2f}%")
        unsupressed_accuracies.append(accuracy)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tests:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, inject_faults=True, suppress_errors=True, error_rate=error_rate)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy with Error Rate (corrected) {error_rate}: {100. * accuracy:.2f}%")
        suppressed_accuracies.append(accuracy)
        
    plt.figure(figsize=(7, 5))
    plt.plot(error_rates, unsupressed_accuracies, marker='o', linestyle='-', color='black', label='uncorrected')
    plt.plot(error_rates, suppressed_accuracies, marker='o', linestyle='--', color='blue', label='corrected')
    plt.xscale('log')
    plt.xlabel("Error Rate")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Pruned Model Accuracy vs. Error Rate")
    plt.show()

model.eval()

# correct = 0
# total = 0
# with torch.no_grad():
#     # clean run
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images, compute_grad=True)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

plot_accuracy_no_correction_vs_correction(model, test_loader)
