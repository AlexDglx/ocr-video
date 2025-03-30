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
        close_points = [current_point]

        # Check for close points to the current one
        j = i + 1
        while j < n:
            next_point = input_list[j]
            distance = np.linalg.norm(current_point - next_point)

            # If the points are close enough, add to the close_points list
            if distance <= threshold:
                close_points.append(next_point)
                # Remove the j-th element as it's now considered
                input_list = np.delete(input_list, j, axis=0)
                n -= 1
            else:
                j += 1

        # If there are close points, calculate their average
        if len(close_points) > 1:
            average_point = np.mean(close_points, axis=0)
            output_list.append(average_point)

        # Move to the next point
        i += 1

    # Convert output list to numpy array
    output_array = np.array(output_list)

    # Remove outliers (e.g., using Z-score or distance-based exclusion)
    if len(output_array) > 0:
        distances = np.linalg.norm(output_array - np.mean(output_array, axis=0), axis=1)
        z_scores = (distances - np.mean(distances)) / np.std(distances)
        output_array = output_array[z_scores < 2.5]  # Keep points within 2.5 std dev

    return output_array

# Example usage:
input_pairs = np.array((
    [[-4.8400000e+02,1.9198622e+00]],
 [[-4.8100000e+02,1.9198622e+00]],
 [[-4.7900000e+02 , 1.9198622e+00]],
 [[ 1.0990000e+03 , 1.8500490e+00]],
 [[ 1.4460000e+03 , 2.0943952e-01]],
 [[ 1.1020000e+03 , 1.8500490e+00]],
 [[ 3.5620000e+03 , 2.4434610e-01]],
 [[ 3.5600000e+03 , 2.4434610e-01]],
 [[ 1.1050000e+03 , 1.8500490e+00]],
 [[ 3.3480000e+03 , 2.4434610e-01]],
 [[ 3.5360000e+03 , 2.0943952e-01]],
 [[ 3.5690000e+03 , 2.4434610e-01]],
 [[ 3.5340000e+03 , 2.0943952e-01]],
 [[ 3.5680000e+03 , 2.7925268e-01]],
 [[ 3.5320000e+03 , 2.0943952e-01]],
 [[ 1.1120000e+03 , 1.8500490e+00]],
 [[ 1.1100000e+03 , 1.8500490e+00]],
 [[ 1.4770000e+03 , 2.4434610e-01]],
 [[ 3.3190000e+03 , 2.0943952e-01]]
))
threshold = 98
output_list = average_close_pairs(input_pairs, threshold)
print(output_list)
