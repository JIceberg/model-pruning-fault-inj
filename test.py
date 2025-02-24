import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os

# set numpy to seed for consistent results
np.random.seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28 * 28  # Flattened image size
hidden_size = 128
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

class PrunedNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, prune_threshold=0.4):
        super(PrunedNN, self).__init__()
        # architecture
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)

        # fault injection
        self.error_rate = 0.1
        self.k = 6
        self.mean_grad = {}
        self.var_grad = {}
        self.num_updates = {}

        # pruning
        self.prune_threshold = prune_threshold
        self.masks = {}

    def forward(self, x, compute_grad=False, inject_faults=False):
        x = x.view(-1, input_size)  # Flatten the image
        x = self.fc1(x)
        if not self.training:
            if compute_grad:
                self.update_running_statistics(x, "fc1")
            else:
                if inject_faults:
                    x = self.bit_flip_fault_inj(x)
                    if "fc1" in self.mean_grad:
                        x = self.threshold_gradients(x, "fc1")
        x = self.relu(x)
        x = self.fc2(x)
        if not self.training:
            if compute_grad:
                self.update_running_statistics(x, "fc2")
            else:
                if inject_faults:
                    x = self.bit_flip_fault_inj(x)
                    if "fc2" in self.mean_grad:
                        x = self.threshold_gradients(x, "fc2")
        return x
    
    def bit_flip_fault_inj(self, output):
        # Flatten tensor for easier manipulation
        flat_output = output.view(-1)

        # Convert float tensor to int representation (IEEE 754)
        float_bits = flat_output.to(torch.float32).cpu().numpy().view(np.uint32)

        # Randomly select bits to flip
        num_elements = flat_output.numel()
        random_bits = np.random.randint(0, 32, size=num_elements, dtype=np.uint32)

        # Create a mask to determine which values to flip
        flip_mask = np.random.rand(num_elements) < self.error_rate

        # Perform bitwise XOR only for selected neurons
        flipped_bits = float_bits ^ (1 << random_bits)
        
        # Ensure numerical stability (avoid NaN, Inf)
        flipped_vals = flipped_bits.view(np.float32)
        valid_mask = np.isfinite(flipped_vals)
        flipped_vals[~valid_mask] = flat_output.cpu().numpy()[~valid_mask]  # Restore original if NaN/Inf

        # Replace only values where flip_mask is True
        float_bits[flip_mask] = flipped_bits[flip_mask]

        # Convert back to PyTorch tensor
        modified_output = torch.tensor(float_bits.view(np.float32), dtype=torch.float32, device=output.device).view(output.shape)

        return modified_output

    def update_running_statistics(self, layer, layer_name):
        layer_unsqeueezed = layer.unsqueeze(1)
        kernel = torch.tensor([-1.0, 1.0, 0.0]).view(1, 1, 3)
        grad_Y = F.conv1d(layer_unsqeueezed, kernel.to(layer.device), padding=1).squeeze(1)

        mean = grad_Y.mean(dim=0).cpu().detach().numpy()
        var = grad_Y.var(dim=0).cpu().detach().numpy()

        if layer_name not in self.mean_grad:
            self.mean_grad[layer_name] = mean
            self.var_grad[layer_name] = var
            self.num_updates[layer_name] = 0
        else:
            self.num_updates[layer_name] += 1
            self.mean_grad[layer_name] += (mean - self.mean_grad[layer_name]) / self.num_updates[layer_name]
            self.var_grad[layer_name] += (var - self.var_grad[layer_name]) / self.num_updates[layer_name]
    
    def threshold_gradients(self, layer, layer_name):
        if self.mean_grad[layer_name] is None or self.var_grad[layer_name] is None:
            return layer
        
        std_grad = np.sqrt(self.var_grad[layer_name])

        mean_grad_tensor = torch.tensor(self.mean_grad[layer_name], device=layer.device)
        std_grad_tensor = torch.tensor(std_grad, device=layer.device)

        lower_bound = mean_grad_tensor - self.k * std_grad_tensor
        upper_bound = mean_grad_tensor + self.k * std_grad_tensor

        layer_unsqeueezed = layer.unsqueeze(1)
        kernel = torch.tensor([-1.0, 1.0, 0.0]).view(1, 1, 3)
        grad_Y = F.conv1d(layer_unsqeueezed, kernel.to(layer.device), padding=1).squeeze(1)

        mask = (grad_Y < lower_bound) | (grad_Y > upper_bound)
        masked_layer = layer.clone()
        masked_layer[mask] = 0 # mean_grad_tensor.expand_as(masked_layer)[mask]

        return masked_layer
    
    def prune_weights(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if "weight" in name:
                    nonzero_weights = param[param != 0].abs()
                    if nonzero_weights.numel() == 0:
                        continue  # Skip if there are no nonzero weights
                    threshold = torch.quantile(nonzero_weights, 1 - self.prune_threshold)
                    mask = (param.abs() >= threshold).float()
                    self.masks[name] = mask
                    param *= mask
    
    def enforce_pruning(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.masks:
                    param *= self.masks[name]   # keep pruned weights at zero

    def count_zero_nonzero_weights(self):
        total_weights = 0
        zero_weights = 0

        for name, param in self.named_parameters():
            if "weight" in name:
                total_weights += param.numel()
                zero_weights += torch.sum(param == 0).item()

        nonzero_weights = total_weights - zero_weights
        zero_percentage = (zero_weights / total_weights) * 100
        nonzero_percentage = (nonzero_weights / total_weights) * 100

        print(f"Total Weights: {total_weights}")
        print(f"Zero Weights: {zero_weights} ({zero_percentage:.2f}%)")
        print(f"Nonzero Weights: {nonzero_weights} ({nonzero_percentage:.2f}%)")

model = PrunedNN(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if os.path.exists("saved_model.pth"):
    model.load_state_dict(torch.load("saved_model.pth", weights_only=True))
else:
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.enforce_pruning()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    fine_tune_iterations = 3

    for _ in range(fine_tune_iterations):
        model.prune_weights()

        model.count_zero_nonzero_weights()

        for epoch in range(num_epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                model.enforce_pruning()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "saved_model.pth")

model.count_zero_nonzero_weights()

model.eval()

correct = 0
total = 0
with torch.no_grad():
    # clean run
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy on Clean Run: {accuracy:.2f}%")

correct = 0
total = 0
with torch.no_grad():
    # faulty run, no correction
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, inject_faults=True)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy with Error Rate {model.error_rate}: {accuracy:.2f}%")

correct = 0
total = 0
with torch.no_grad():
    # compute gradient statistics
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, compute_grad=True)

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, inject_faults=True)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy with Error Correction: {accuracy:.2f}%")
