import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MNISTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, k=6, gamma=8):
        super(MNISTClassifier, self).__init__()
        # architecture
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)

        # fault injection
        self.k = k
        self.gamma = gamma
        self.mean_grad = {}
        self.var_grad = {}
        self.num_updates = {}

    def forward(self, x, compute_grad=False, inject_faults=False, suppress_errors=False, error_rate=0.1):
        x = x.view(-1, self.input_size)  # Flatten the image
        x = self.fc1(x)
        if not self.training:
            if compute_grad:
                self.update_running_statistics(x, "fc1")
            else:
                if inject_faults:
                    x = self.bit_flip_fault_inj(x, error_rate=error_rate)
                    if suppress_errors:
                        if "fc1" in self.mean_grad:
                            x = self.threshold_gradients(x, "fc1")
        x = self.relu(x)
        x = self.fc2(x)
        if not self.training:
            if compute_grad:
                self.update_running_statistics(x, "fc2")
            else:
                if inject_faults:
                    x = self.bit_flip_fault_inj(x, error_rate=error_rate)
                    if suppress_errors:
                        if "fc2" in self.mean_grad:
                            x = self.threshold_gradients(x, "fc2")
        return x
    
    def bit_flip_fault_inj(self, output, error_rate=0.1):
        # Flatten tensor for easier manipulation
        flat_output = output.view(-1)

        # Convert float tensor to int representation (IEEE 754)
        float_bits = flat_output.to(torch.float32).cpu().numpy().view(np.uint32)

        # Randomly select bits to flip
        num_elements = flat_output.numel()
        random_bits = np.random.randint(0, 32, size=num_elements, dtype=np.uint32)

        # Create a mask to determine which values to flip
        flip_mask = np.random.rand(num_elements) < error_rate

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
        new_layer = layer.clone()
        for i in range(1, layer.size(0)):
            new_layer[i, mask[i]] = mean_grad_tensor[mask[i]] + layer[i, mask[i]]

        return new_layer