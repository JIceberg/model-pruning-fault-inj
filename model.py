import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def nan_checker(x):
    nan_check = torch.isnan(x)
    inf_check = torch.isinf(x)
    if torch.sum(nan_check) or torch.sum(inf_check):
        x = x.masked_fill_(nan_check,0)
        x = x.masked_fill_(inf_check,0)
    return x  

def flip_bits(A, error_rate=1e-4):
    # Flatten tensor for easier manipulation
    flat_output = A.view(-1)

    # Convert float tensor to int representation (IEEE 754)
    float_bits = flat_output.to(torch.float32).cpu().numpy().view(np.uint32)

    # Randomly select bits to flip
    num_elements = flat_output.numel()
    random_bits = np.random.randint(0, 32, size=num_elements, dtype=np.uint32)

    # Create a mask to determine which values to flip
    flip_mask = np.random.rand(num_elements) < error_rate

    # Perform bitwise XOR only for selected neurons
    flipped_bits = float_bits ^ (1 << random_bits)

    # Replace only values where flip_mask is True
    float_bits[flip_mask] = flipped_bits[flip_mask]

    # Convert back to PyTorch tensor
    modified_output = torch.tensor(float_bits.view(np.float32), dtype=torch.float32, device=A.device).view(A.shape)
    return modified_output

class Correction_Module_dense(nn.Module):
    def __init__(self):
        super(Correction_Module_dense, self).__init__()
        self.mean_grad = {}
        self.var_grad = {}
        self.num_updates = {}
        self.k = 4

    def get_gradient(self, x):
        convolution_nn = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=3,padding='same',padding_mode='circular')
        convolution_nn.weight.requires_grad = False 
        convolution_nn.bias.requires_grad = False 
        convolution_nn.weight[0] = torch.tensor([-1,1,0],dtype=torch.float)
        convolution_nn.bias[0] = torch.tensor([0],dtype=torch.float)
        convolution_nn = convolution_nn.to(x.device)

        grad = []
        for batchind in range(0, x.shape[0]):
            gradind = convolution_nn(x[batchind,:].reshape(1,1,x.shape[1])).reshape(1,x.shape[1])
            grad.append(gradind)

        return torch.stack(grad, dim=0).view(-1, x.shape[1])
    
    def compute_grad(self, x, layer_name):
        if layer_name not in self.num_updates:
            self.num_updates[layer_name] = 0
        self.num_updates[layer_name] += 1

        x = nan_checker(x)
        grad_Y = self.get_gradient(x)
        mean = grad_Y.mean(dim=0).cpu().detach().numpy()
        var = grad_Y.var(dim=0).cpu().detach().numpy()

        if layer_name not in self.mean_grad:
            self.mean_grad[layer_name] = mean
            self.var_grad[layer_name] = var
        else:
            self.mean_grad[layer_name] += (mean - self.mean_grad[layer_name]) / self.num_updates[layer_name]
            self.var_grad[layer_name] += (var - self.var_grad[layer_name]) / self.num_updates[layer_name]

    def forward(self, input, output, layer_func, layer_name):
        if self.mean_grad[layer_name] is None or self.var_grad[layer_name] is None:
            return output

        batch_size, num_neurons = output.shape
        output = nan_checker(output)
        
        std_grad = np.sqrt(self.var_grad[layer_name])

        mean_grad_tensor = torch.tensor(self.mean_grad[layer_name], device=output.device)
        std_grad_tensor = torch.tensor(std_grad, device=output.device)

        lower_bound = mean_grad_tensor - self.k * std_grad_tensor
        upper_bound = mean_grad_tensor + self.k * std_grad_tensor

        grad_Y = self.get_gradient(output)
        mask = (grad_Y < lower_bound) | (grad_Y > upper_bound)

        new_output = output.clone()
        new_output[mask] = 0

        # ground_truth = layer_func(input)
        # print("old layer:", output[0, mask[0]])
        # print("new layer:", new_output[0, mask[0]])
        # print("true layer:", ground_truth[0, mask[0]])

        return new_output
    
class Correction_Module_conv(nn.Module):
    def __init__(self):
        super(Correction_Module_conv, self).__init__()
        self.mean_grad = {}
        self.var_grad = {}
        self.num_updates = {}
        self.k = 6

    def get_gradient(self, x):
        convolution_nn = nn.Conv1d(in_channels=x.shape[2],out_channels=x.shape[2],kernel_size=3,
            padding='same',padding_mode='circular')
        convolution_nn.weight.requires_grad = False 
        convolution_nn.bias.requires_grad = False 
        convolution_nn.weight[0] = torch.tensor([-1,1,0],dtype=torch.float)
        convolution_nn.bias[0] = torch.tensor([0],dtype=torch.float)
        convolution_nn = convolution_nn.to(x.device)

        grads = []
        for batchind in range(0, x.shape[0]):
            outtemp = torch.swapaxes(x[batchind,:,:,:], 0, 2)
            tempout_test = convolution_nn(outtemp)
            grad = torch.swapaxes(tempout_test, 0, 2)
            grads.append(grad)
        
        grads = torch.stack(grads, dim=0)
        return grads
    
    def compute_grad(self, x, layer_name):
        if layer_name not in self.num_updates:
            self.num_updates[layer_name] = 0
        self.num_updates[layer_name] += 1

        x = nan_checker(x)

        grad_Y = self.get_gradient(x)
        
        mean = grad_Y.mean(dim=0).cpu().detach().numpy()
        var = grad_Y.var(dim=0).cpu().detach().numpy()
        if layer_name not in self.mean_grad:
            self.mean_grad[layer_name] = mean
            self.var_grad[layer_name] = var
        else:
            self.mean_grad[layer_name] += (mean - self.mean_grad[layer_name]) / self.num_updates[layer_name]
            self.var_grad[layer_name] += (var - self.var_grad[layer_name]) / self.num_updates[layer_name]

    def forward(self, input, output, layer_func, layer_name):
        if self.mean_grad[layer_name] is None or self.var_grad[layer_name] is None:
            return output

        output = nan_checker(output)
        
        std_grad = np.sqrt(self.var_grad[layer_name])

        mean_grad_tensor = torch.tensor(self.mean_grad[layer_name], device=output.device)
        std_grad_tensor = torch.tensor(std_grad, device=output.device)

        lower_bound = mean_grad_tensor - self.k * std_grad_tensor
        upper_bound = mean_grad_tensor + self.k * std_grad_tensor

        grad_Y = self.get_gradient(output)
        mask = (grad_Y < lower_bound) | (grad_Y > upper_bound)

        new_output = output.clone()
        new_output[mask] = 0

        # ground_truth = layer_func(input)
        # print("old layer:", output[0, mask[0]])
        # print("new layer:", new_output[0, mask[0]])
        # print("true layer:", ground_truth[0, mask[0]])

        return new_output
    
class Correction_Module_dense_checksum(nn.Module):
    def __init__(self, k=2, atol=1e-3):
        super(Correction_Module_dense_checksum, self).__init__()
        self.k = k
        self.atol = atol
    
    def block_checksum_matrix_left(self, A):
        m, n = A.shape
        w = math.ceil(m / self.k)
        pad_rows = w * self.k - m

        if pad_rows > 0:
            pad = A.new_zeros((pad_rows, n))
            A_padded = torch.cat([A, pad], dim=0)
        else:
            A_padded = A

        AC = A_padded.view(w, self.k, n).sum(dim=1)
        return AC

    def block_checksum_matrix_right(self, B):
        m, n = B.shape
        w = math.ceil(n / self.k)
        pad_cols = w * self.k - n

        if pad_cols > 0:
            pad = B.new_zeros((m, pad_cols))
            B_padded = torch.cat([B, pad], dim=1)
        else:
            B_padded = B

        BC = B_padded.view(m, w, self.k).sum(dim=2)
        return BC
    
    def block_checksum_2d(self, C):
        left = self.block_checksum_matrix_left(C)
        return self.block_checksum_matrix_right(left)
    
    def forward(self, A, B, C_faulty, error_rate=0.0):
        """
        Detects and recomputes erroneous blocks in C using 2D block checksums.
        """
        m, n = C_faulty.shape
        k = self.k

        # Compute 2D block checksums for B and A
        AC = self.block_checksum_matrix_left(A)         # (ceil(m/k), n)
        BC = self.block_checksum_matrix_left(B)         # (ceil(p/k), n)
        CC_check = F.linear(BC, AC)

        # Actual checksum of the faulty matrix
        CC_actual = self.block_checksum_2d(C_faulty)    # shape: (h, w)

        # Identify mismatches
        diff = ~torch.isclose(CC_actual, CC_check, rtol=1e-4, atol=self.atol)
        error_blocks = torch.nonzero(diff, as_tuple=False)

        for bi, bj in error_blocks:
            row_start = bi * k
            row_end = min((bi + 1) * k, m)
            col_start = bj * k
            col_end = min((bj + 1) * k, n)

            # Select submatrices
            B_block = B[row_start:row_end, :]         # shape (k, n)
            A_block = A[col_start:col_end, :]         # shape (k, n)

            # Recompute block of C
            C_block = F.linear(B_block, A_block)      # shape: (k, k)
            C_block = flip_bits(C_block, error_rate=error_rate)
            C_faulty[row_start:row_end, col_start:col_end] = C_block

        return C_faulty

class MNISTClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(MNISTClassifier, self).__init__()
        self.input_size = input_size

        # architecture
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.relu = nn.ReLU()
        
        self.corr_dense = Correction_Module_dense_checksum(k=4, atol=1e-4)

    def forward(self, x, compute_grad=False, inject_faults=False, suppress_errors=False, error_rate=0.0):        
        fc1_input = x.view(-1, self.input_size)
        fc1_output = self.fc1(fc1_input)
        if inject_faults:
            fc1_output = flip_bits(fc1_output, error_rate=error_rate)
        if compute_grad:
            self.corr_dense.compute_grad(fc1_output, "fc1")
        if suppress_errors:
            # fc1_output = self.corr_dense(fc1_input, fc1_output, self.fc1, "fc1")
            fc1_output = self.corr_dense(self.fc1.weight, fc1_input, fc1_output, error_rate=error_rate)
        
        fc2_input = self.relu(fc1_output)
        fc2_output = self.fc2(fc2_input)
        if inject_faults:
            fc2_output = flip_bits(fc2_output, error_rate=error_rate)
        if compute_grad:
            self.corr_dense.compute_grad(fc2_output, "fc2")
        if suppress_errors:
            # fc2_output = self.corr_dense(fc2_input, fc2_output, self.fc2, "fc2")
            fc2_output = self.corr_dense(self.fc2.weight, fc2_input, fc2_output, error_rate=error_rate)
        
        return fc2_output