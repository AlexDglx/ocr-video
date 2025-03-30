import numpy as np

def average_close_pairs(input_list, threshold):
    # Initialize output list
    output_list = []
    n = len(input_list)

    # Iterate through the list
    i = 0
    while i < n:
        # Get the current point as a numpy array
        current_point = input_list[i]

        # Check if the next point is close enough to the current one
        j = i + 1
        while j < n:
            next_point = input_list[j]

            # Calculate the distance between current and next point
            distance = np.linalg.norm(current_point - next_point)

            # If the points are close enough, average them and remove the next point
            if distance <= threshold:
                current_point = (current_point + next_point) / 2
                # Remove the j-th element from the list as it's been averaged
                input_list = np.delete(input_list, j, axis=0)
                n -= 1
            else:
                j += 1
        
        # Append the averaged point to the output list
        output_list.append(current_point)
        i += 1

    return np.array(output_list)

# Example usage:
input_pairs = np.array ([[[1.5600000e+02, 1.7104226e+00]], 
                [[1.5800000e+02, 1.7104226e+00]],
                [[5.9100000e+02, 1.7802358e+00]],
                [[1.6790000e+03, 1.0471976e-01]],
                [[8.4000000e+02, 3.8397244e-01]],
                [[1.6600000e+03, 6.9813170e-02]],
                [[8.4300000e+02, 3.8397244e-01]],
                [[8.3300000e+02, 3.4906584e-01]],
                [[5.4800000e+02, 1.8151424e+00]]
               ])
threshold = 95
output_list = average_close_pairs(input_pairs, threshold)
print(output_list)
