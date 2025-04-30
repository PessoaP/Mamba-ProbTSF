import numpy as np
import os
def original_shape(array, shape=(4,92,321)):
    reshaped = array.reshape(-1, shape[2], array.shape[-1])
    return reshaped.transpose(0, 2, 1)

def save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, 
                           lower_bounds=None, upper_bounds=None, save_bounds=False, original_shape=original_shape):
    os.makedirs(folder_path, exist_ok=True)

    medians_array = original_shape(np.array(medians))
    np.save(os.path.join(folder_path, 'medians.npy'), medians_array)

    means_array = original_shape(np.array(means))
    np.save(os.path.join(folder_path, 'means.npy'), means_array)

    within_arrays = [original_shape(np.array(w)) for w in within_intervals]
    for k, arr in enumerate(within_arrays):
        np.save(os.path.join(folder_path, f'sigma{k+1}.npy'), arr)
        print(f'Within {confidence[k]*100:.1f}% interval:', arr.mean())

    if save_bounds and lower_bounds is not None and upper_bounds is not None:
        lower_arrays = [original_shape(np.array(lb)) for lb in lower_bounds]
        upper_arrays = [original_shape(np.array(ub)) for ub in upper_bounds]
        for k in range(3):
            np.save(os.path.join(folder_path, f'sigma{k+1}_lower.npy'), lower_arrays[k])
            np.save(os.path.join(folder_path, f'sigma{k+1}_upper.npy'), upper_arrays[k])
