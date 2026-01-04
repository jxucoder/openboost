"""All CUDA kernels - defined at module level to avoid recompilation."""

import numpy as np
from numba import cuda

HIST_BLOCK_SIZE = 256
MAX_BINS = 256


@cuda.jit
def histogram_kernel(
    X_binned,      # (n_features, n_samples) uint8
    gradients,     # (n_samples,) float32
    sample_nodes,  # (n_samples,) int32 - which node each sample belongs to
    node_start,    # int32 - first node index at this level
    n_nodes,       # int32 - number of nodes at this level
    histograms,    # (n_nodes, n_features, MAX_BINS, 2) float32 - [sum_grad, count]
):
    """Build histograms for all nodes at a level using shared memory."""
    # Shared memory for local histogram: (MAX_BINS, 2) per feature processed
    local_hist = cuda.shared.array((MAX_BINS, 2), dtype=np.float32)
    
    tid = cuda.threadIdx.x
    feature_idx = cuda.blockIdx.y
    node_offset = cuda.blockIdx.z  # Which node (0 to n_nodes-1)
    node_idx = node_start + node_offset
    
    n_samples = gradients.shape[0]
    
    # Initialize shared memory
    if tid < MAX_BINS:
        local_hist[tid, 0] = 0.0  # sum_grad
        local_hist[tid, 1] = 0.0  # count
    cuda.syncthreads()
    
    # Process samples in grid-stride loop
    idx = cuda.blockIdx.x * cuda.blockDim.x + tid
    stride = cuda.gridDim.x * cuda.blockDim.x
    
    while idx < n_samples:
        if sample_nodes[idx] == node_idx:
            bin_idx = X_binned[feature_idx, idx]
            grad = gradients[idx]
            cuda.atomic.add(local_hist, (bin_idx, 0), grad)
            cuda.atomic.add(local_hist, (bin_idx, 1), 1.0)
        idx += stride
    
    cuda.syncthreads()
    
    # Reduce to global histogram
    if tid < MAX_BINS:
        cuda.atomic.add(histograms, (node_offset, feature_idx, tid, 0), local_hist[tid, 0])
        cuda.atomic.add(histograms, (node_offset, feature_idx, tid, 1), local_hist[tid, 1])


@cuda.jit
def find_best_split_kernel(
    histograms,     # (n_nodes, n_features, MAX_BINS, 2) float32
    node_sum_grad,  # (n_nodes,) float32 - total gradient sum per node
    node_count,     # (n_nodes,) float32 - total count per node
    best_gain,      # (n_nodes,) float32 - output
    best_feature,   # (n_nodes,) int32 - output
    best_bin,       # (n_nodes,) int32 - output
    min_samples_leaf,  # int32
):
    """Find best split for each node. One block per node."""
    node_idx = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    
    n_features = histograms.shape[1]
    
    # Shared memory for reduction
    shared_gain = cuda.shared.array(HIST_BLOCK_SIZE, dtype=np.float32)
    shared_feature = cuda.shared.array(HIST_BLOCK_SIZE, dtype=np.int32)
    shared_bin = cuda.shared.array(HIST_BLOCK_SIZE, dtype=np.int32)
    
    total_grad = node_sum_grad[node_idx]
    total_count = node_count[node_idx]
    
    my_best_gain = -1e10
    my_best_feature = -1
    my_best_bin = -1
    
    # Each thread processes multiple (feature, bin) combinations
    for work_idx in range(tid, n_features * MAX_BINS, cuda.blockDim.x):
        f = work_idx // MAX_BINS
        b = work_idx % MAX_BINS
        
        # Compute cumulative sum up to bin b for feature f
        left_grad = 0.0
        left_count = 0.0
        for i in range(b + 1):
            left_grad += histograms[node_idx, f, i, 0]
            left_count += histograms[node_idx, f, i, 1]
        
        right_grad = total_grad - left_grad
        right_count = total_count - left_count
        
        # Check min_samples_leaf
        if left_count >= min_samples_leaf and right_count >= min_samples_leaf:
            # MSE gain: (left_grad^2 / left_count) + (right_grad^2 / right_count)
            gain = 0.0
            if left_count > 0:
                gain += (left_grad * left_grad) / left_count
            if right_count > 0:
                gain += (right_grad * right_grad) / right_count
            
            if gain > my_best_gain:
                my_best_gain = gain
                my_best_feature = f
                my_best_bin = b
    
    shared_gain[tid] = my_best_gain
    shared_feature[tid] = my_best_feature
    shared_bin[tid] = my_best_bin
    cuda.syncthreads()
    
    # Tree reduction to find best
    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            if shared_gain[tid + s] > shared_gain[tid]:
                shared_gain[tid] = shared_gain[tid + s]
                shared_feature[tid] = shared_feature[tid + s]
                shared_bin[tid] = shared_bin[tid + s]
        cuda.syncthreads()
        s //= 2
    
    if tid == 0:
        best_gain[node_idx] = shared_gain[0]
        best_feature[node_idx] = shared_feature[0]
        best_bin[node_idx] = shared_bin[0]


@cuda.jit
def update_sample_nodes_kernel(
    X_binned,       # (n_features, n_samples) uint8
    sample_nodes,   # (n_samples,) int32 - updated in place
    node_start,     # int32 - first node at current level
    n_nodes,        # int32 - nodes at current level
    split_features, # (n_nodes,) int32
    split_bins,     # (n_nodes,) int32
    is_leaf,        # (n_nodes,) bool - whether node is leaf
):
    """Update sample node assignments after splits."""
    idx = cuda.grid(1)
    if idx >= sample_nodes.shape[0]:
        return
    
    node_idx = sample_nodes[idx]
    node_offset = node_idx - node_start
    
    # Check if this sample's node is being split at this level
    if node_offset < 0 or node_offset >= n_nodes:
        return
    
    if is_leaf[node_offset]:
        return
    
    feature = split_features[node_offset]
    split_bin = split_bins[node_offset]
    
    # Left child = 2 * node_idx + 1, Right child = 2 * node_idx + 2
    if X_binned[feature, idx] <= split_bin:
        sample_nodes[idx] = 2 * node_idx + 1
    else:
        sample_nodes[idx] = 2 * node_idx + 2


@cuda.jit
def predict_kernel(
    X_binned,       # (n_features, n_samples) uint8
    tree_features,  # (max_nodes,) int32
    tree_bins,      # (max_nodes,) int32
    tree_values,    # (max_nodes,) float32
    tree_is_leaf,   # (max_nodes,) bool
    predictions,    # (n_samples,) float32 - output, accumulated
    learning_rate,  # float32
):
    """Traverse tree for each sample and accumulate predictions."""
    idx = cuda.grid(1)
    if idx >= predictions.shape[0]:
        return
    
    node = 0
    while not tree_is_leaf[node]:
        feature = tree_features[node]
        if X_binned[feature, idx] <= tree_bins[node]:
            node = 2 * node + 1
        else:
            node = 2 * node + 2
    
    predictions[idx] += learning_rate * tree_values[node]

