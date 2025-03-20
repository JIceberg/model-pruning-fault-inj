import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MNISTClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, k=5):
        super(MNISTClassifier, self).__init__()
        # architecture
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=False)

        # fault injection
        self.k = k
        self.mean_grad = {}
        self.var_grad = {}
        self.num_updates = {}

    def forward(self, x, compute_grad=False, inject_faults=False, suppress_errors=False, error_rate=0.1, zeroing=False):
        fc1_input = x.view(-1, self.input_size)  # Flatten the image
        fc1_output = self.fc1(fc1_input)
        if not self.training:
            if compute_grad:
                self.update_running_statistics(fc1_output, "fc1")
            else:
                if inject_faults:
                    fc1_output = self.bit_flip_fault_inj(fc1_output, error_rate=error_rate)
                    if suppress_errors:
                        if "fc1" in self.mean_grad:
                            fc1_output = self.threshold_gradients(
                                fc1_input,
                                fc1_output,
                                self.fc1,
                                "fc1",
                                error_rate=error_rate,
                                zeroing=zeroing)
        fc2_input = self.relu(fc1_output)
        fc2_output = self.fc2(fc2_input)
        if not self.training:
            if compute_grad:
                self.update_running_statistics(fc2_output, "fc2")
            else:
                if inject_faults:
                    fc2_output = self.bit_flip_fault_inj(fc2_output, error_rate=error_rate)
                    if suppress_errors:
                        if "fc2" in self.mean_grad:
                            fc2_output = self.threshold_gradients(
                                fc2_input,
                                fc2_output,
                                self.fc2,
                                "fc2",
                                error_rate=error_rate,
                                zeroing=zeroing)
        return fc2_output
    
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
        if layer_name not in self.num_updates:
            self.num_updates[layer_name] = 0
        self.num_updates[layer_name] += 1

        layer_unsqeueezed = layer.unsqueeze(1)
        kernel = torch.tensor([-1.0, 1.0, 0.0]).view(1, 1, 3)
        grad_Y = F.conv1d(layer_unsqeueezed, kernel.to(layer.device), padding=1).squeeze(1)
        
        mean = grad_Y.mean(dim=0).cpu().detach().numpy()
        var = grad_Y.var(dim=0).cpu().detach().numpy()
        if layer_name not in self.mean_grad:
            self.mean_grad[layer_name] = mean
            self.var_grad[layer_name] = var
        else:
            self.mean_grad[layer_name] += (mean - self.mean_grad[layer_name]) / self.num_updates[layer_name]
            self.var_grad[layer_name] += (var - self.var_grad[layer_name]) / self.num_updates[layer_name]
    
    def threshold_gradients(self, input, output, layer_func, layer_name, error_rate=0.1, zeroing=False):
        if self.mean_grad[layer_name] is None or self.var_grad[layer_name] is None:
            return output
        
        batch_size, num_neurons = output.shape
        
        std_grad = np.sqrt(self.var_grad[layer_name])

        mean_grad_tensor = torch.tensor(self.mean_grad[layer_name], device=output.device)
        std_grad_tensor = torch.tensor(std_grad, device=output.device)

        lower_bound = mean_grad_tensor - self.k * std_grad_tensor
        upper_bound = mean_grad_tensor + self.k * std_grad_tensor

        layer_unsqeueezed = output.unsqueeze(1)
        left_kernel = torch.tensor([-1.0, 1.0, 0.0]).view(1, 1, 3)
        right_kernel = torch.tensor([0.0, 1.0, -1.0]).view(1, 1, 3)
        left_grad = F.conv1d(layer_unsqeueezed, left_kernel.to(output.device), padding=1).squeeze(1)
        right_grad = F.conv1d(layer_unsqeueezed, right_kernel.to(output.device), padding=1).squeeze(1)

        mask1 = ((left_grad < lower_bound) | (left_grad > upper_bound))
        mask2 = ((right_grad < lower_bound) | (right_grad > upper_bound))
        mask = mask1 & mask2

        new_output = output.clone()
        if zeroing:
            new_output[mask] = 0
        else:
            left_values = torch.roll(output, 1, 1)
            left_values[:, 0] = 0
            recomputed_values = left_values + mean_grad_tensor
            new_output[mask] = recomputed_values[mask]

        ground_truth = layer_func(input)

        # print("old layer:", output[0, mask[0]])
        # print("new layer:", new_output[0, mask[0]])
        # print("true layer:", ground_truth[0, mask[0]])

        return new_output