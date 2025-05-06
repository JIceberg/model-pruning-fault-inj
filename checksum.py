import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def row_checksum(A):
    m, n = A.shape
    row_sum = torch.sum(A, dim=1).view(m, 1)
    return torch.cat((A, row_sum), dim=1)
    
def col_checksum(B):
    m, n = B.shape
    col_sum = torch.sum(B, dim=0).view(1, n)
    return torch.cat((B, col_sum), dim=0)

def block_checksum_matrix_left(A, k=2):
    m, n = A.shape
    w = math.ceil(m / k)
    pad_rows = w * k - m

    if pad_rows > 0:
        pad = A.new_zeros((pad_rows, n))
        A_padded = torch.cat([A, pad], dim=0)
    else:
        A_padded = A
    AC = A_padded.view(w, k, n).sum(dim=1)
    return AC

def block_checksum_matrix_right(B, k=2):
    m, n = B.shape
    w = math.ceil(n / k)
    pad_cols = w * k - n

    if pad_cols > 0:
        pad = B.new_zeros((m, pad_cols))
        B_padded = torch.cat([B, pad], dim=1)
    else:
        B_padded = B
    BC = B_padded.view(m, w, k).sum(dim=2)
    return BC

def block_checksum_2d(C, k):
    # First reduce row-wise (vertical blocks)
    left = block_checksum_matrix_left(C, k)  # shape: (h, n)
    # Then reduce column-wise (horizontal blocks)
    return block_checksum_matrix_right(left, k)  # shape: (h, w)

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
    affected_indices = np.where(flip_mask)[0]
    print("Values before flipping:")
    for idx in affected_indices:
        original_value = float_bits[idx].view(np.float32)
        print(f"Index: {idx}, Original Value: {original_value}")

    # Perform bitwise XOR only for selected neurons
    flipped_bits = float_bits ^ (1 << random_bits)

    # Replace only values where flip_mask is True
    float_bits[flip_mask] = flipped_bits[flip_mask]
    print("Values after flipping:")
    for idx in affected_indices:
        flipped_value = float_bits[idx].view(np.float32)
        print(f"Index: {idx}, Flipped Value: {flipped_value}")

    # Convert back to PyTorch tensor
    modified_output = torch.tensor(float_bits.view(np.float32), dtype=torch.float32, device=A.device).view(A.shape)
    return modified_output

def recompute_faulty_blocks_2d(A, B, C_faulty, k=4, atol=1e-3):
    """
    Detects and recomputes erroneous blocks in C using 2D block checksums.
    """
    m, n = C_faulty.shape

    # Compute 2D block checksums for B and A
    AC = block_checksum_matrix_left(A, k)         # (ceil(m/k), n)
    BC = block_checksum_matrix_left(B, k)         # (ceil(p/k), n)
    CC_check = F.linear(BC, AC)             # (ceil(p/k), ceil(m/k))

    # Actual checksum of the faulty matrix
    CC_actual = block_checksum_2d(C_faulty, k)    # shape: (h, w)

    # Identify mismatches
    diff = ~torch.isclose(CC_actual, CC_check, rtol=1e-4, atol=atol)
    error_blocks = torch.nonzero(diff, as_tuple=False)

    print(f"Detected {len(error_blocks)} faulty blocks")

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
        C_block = flip_bits(C_block)    # represent error on linear computations
        C_faulty[row_start:row_end, col_start:col_end] = C_block
        print(f"Recomputed block ({bi},{bj})")

    return C_faulty


A = torch.rand(128, 10)
B = torch.rand(64, 10)
C = F.linear(B, A)
# C = flip_bits(C)

'''
block checksum method
'''

k = 4
AC = block_checksum_matrix_left(A, k)
BC = block_checksum_matrix_left(B, k)
CC_check = F.linear(BC, AC)
print(CC_check.shape)

CC = block_checksum_matrix_right(block_checksum_matrix_left(C, k), k)
if not torch.allclose(CC, CC_check):
    print("Error(s) Detected")
    diff_indices = torch.nonzero(~torch.isclose(CC, CC_check), as_tuple=True)
    for i, j in zip(*diff_indices):
        row_start = i * k
        row_end = min((i + 1) * k, C.shape[0])
        col_start = j * k
        col_end = min((j + 1) * k, C.shape[1])
        block = C[row_start:row_end, col_start:col_end]
        block_sum = block.sum().item()
        expected_sum = CC_check[i, j].item()
        print(f"[Block {i}, {j}] Sum = {block_sum:.6f}, Expected = {expected_sum:.6f}")
        
    C = recompute_faulty_blocks_2d(A, B, C, k=k)
    CC = block_checksum_matrix_right(block_checksum_matrix_left(C, k), k)
    print(torch.allclose(CC, CC_check))
else:
    print("No errors!")
