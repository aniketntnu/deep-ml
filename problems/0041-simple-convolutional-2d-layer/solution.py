import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # Step 1: Add zero padding around input
    # Example: padding=1 → add 1 layer of zeros on all sides
    padded = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')

    # Step 2: Compute output dimensions
    # Formula: ((input + 2*padding - kernel) // stride) + 1
    out_height = ((input_height + 2*padding - kernel_height) // stride) + 1
    out_width  = ((input_width  + 2*padding - kernel_width)  // stride) + 1

    # Step 3: Initialize output matrix
    output_matrix = np.zeros((out_height, out_width))

    # Step 4: Slide kernel over input
    for i in range(out_height):
        for j in range(out_width):

            # Find top-left corner of current window
            start_i = i * stride
            start_j = j * stride

            # Extract region same size as kernel
            region = padded[start_i:start_i + kernel_height,
                            start_j:start_j + kernel_width]

            # Example:
            # region = [[1,2],
            #           [5,6]]
            # kernel = [[1,0],
            #           [-1,1]]

            # Step 5: Element-wise multiply and sum
            output_matrix[i, j] = np.sum(region * kernel)

    return output_matrix