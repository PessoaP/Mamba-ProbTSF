# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import os
from tqdm import tqdm
import argparse
import gc

from gluonts_utils import original_shape,save_forecast_outputs

from gluonts.dataset.repository import get_dataset, dataset_names
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions


# Define parser
parser = argparse.ArgumentParser()

# Add arguments with default values
parser.add_argument('--folder', type=str, default='ECL_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0')#'Sines_se_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0')
parser.add_argument('--prediction_length', type=int, default=96)
parser.add_argument('--past_length', type=int, default=96)
parser.add_argument('--mamba_batch_size', type=int, default=16)
parser.add_argument('--Method', type=str, default='ARIMA')
#save_bounds = True  # <-- Set True to also save lower/upper bounds
parser.add_argument('--save_bounds', type=bool, default=True)


# Parse args
args = parser.parse_args()

# Assign variables
folder = 'results_prob/'+ args.folder
prediction_length = args.prediction_length
past_length = args.past_length
mamba_batch_size = args.mamba_batch_size
Method = args.Method
save_bounds = args.save_bounds


# Now you can use them normally
print(f"Folder: {folder}")
print(f"Past length: {past_length}")
print(f"Prediction length: {prediction_length}")
print(f"Method: {Method}")


# %%
#Loads same training and testing dataset as used in Mamba-ProbTSF, but in gluonts format.
#This requires one to have run the Mamba-ProbTSF, save exactly how Mamba-ProbTSF separated the test and training data and use the same training data.
past = np.load(folder+'/input.npy')
fut  = np.load(folder+'/trues.npy')
test_shape = past.shape

past = np.transpose(past, (0, 2, 1)).reshape(-1,past_length)
fut  = np.transpose(fut , (0, 2, 1)).reshape(-1,prediction_length)

test_dataset = np.hstack((past,fut))
print(test_shape)
del past
del fut

# %%
# def original_shape(array, shape=test_shape):
#     reshaped = array.reshape(-1, shape[2], array.shape[-1])
#     return reshaped.transpose(0, 2, 1)

# def save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, 
#                            lower_bounds=None, upper_bounds=None, save_bounds=save_bounds,original_shape=original_shape):
#     os.makedirs(folder_path, exist_ok=True)

#     medians_array = np.array(medians)
#     np.save(os.path.join(folder_path, 'medians.npy'), original_shape(medians_array))
#     print('Median avg error:', np.abs(medians_array - np.array(futures)).mean(axis=0))

#     means_array = np.array(means)
#     np.save(os.path.join(folder_path, 'means.npy'), original_shape(means_array))
#     print('Mean avg error:', np.abs(means_array - np.array(futures)).mean(axis=0))

#     for k in range(3):
#         np.save(os.path.join(folder_path, f'sigma{k+1}.npy'), original_shape(np.array(within_intervals[k])))
#         print(f'Within {confidence[k]*100:.1f}% interval:', np.array(within_intervals[k]).mean())

#         if save_bounds and lower_bounds is not None and upper_bounds is not None:
#             np.save(os.path.join(folder_path, f'sigma{k+1}_lower.npy'), original_shape(np.array(lower_bounds[k])))
#             np.save(os.path.join(folder_path, f'sigma{k+1}_upper.npy'), original_shape(np.array(upper_bounds[k])))


perbatch = test_shape[-1]
batch_size = mamba_batch_size * perbatch
print(f"Batch size: {batch_size}")
# %%
freq = "1H"
start = pd.Period("01-01-2019", freq=freq)  # Just part of gluonts test, nothing to be attributed from this

# # %%
# # train dataset: 
# train_ds = ListDataset(
#     [{"target": x, "start": start} for x in custom_dataset],
#     freq=freq,
# )

# test dataset: 
#This is necessary bc ARIMA as implemeented doesnt do this automatically
center = test_dataset.mean(axis=1).reshape(-1,1)
width = test_dataset.std(axis=1).reshape(-1,1)
test_ds = ListDataset(
    [{"target": x, "start": start} for x in ((test_dataset-center)/width)], freq=freq
)


if Method == 'ARIMA':
    from gluonts.ext.r_forecast import RForecastPredictor
    estimator =  RForecastPredictor(
        freq=freq,
        method_name="arima",
        prediction_length=prediction_length,
    )
    predictor= estimator.predict(test_ds)
else:
    print("this code expects ARIMA")
# elif Method == 'FeedForward':


# --- Configuration ---
confidence = np.array([0.683, 0.955, 0.998])
quantiles = (1 + np.array([[-1], [1]]) * confidence).T / 2  # shape (3,2)

# Output folder
folder_path = os.path.join(folder, 'gluonts', Method)
os.makedirs(folder_path, exist_ok=True)



# --- Main loop ---
means = np.empty((batch_size//4,prediction_length))#[]
medians = np.empty_like(means)#[]
futures = np.empty_like(means)
within_intervals = [np.empty_like(means) for _ in range(3)]
lower_bounds = [np.empty_like(means) for _ in range(3)] if save_bounds else None
upper_bounds = [np.empty_like(means) for _ in range(3)] if save_bounds else None


for idx in tqdm(range(batch_size//4)):
    gc.collect()
    if Method == 'ARIMA':
        forecast = next(predictor)
        
        tss = test_ds[idx]['target']
        fut = tss[-prediction_length:] #standardized
        
        m,s = center[idx],width[idx]

        means[idx] = forecast.mean * s + m
        medians[idx] = forecast.median * s + m
        futures[idx] = fut
        for k in range(3):
            lower = forecast.quantile(quantiles[k, 0])
            upper = forecast.quantile(quantiles[k, 1])
            within = np.logical_and(fut > lower, fut < upper)

            within_intervals[k][idx] = within #calculated on standardized

            if save_bounds:
                lower_bounds[k][idx] = lower  * s + m
                upper_bounds[k][idx] = upper  * s + m
                

    if (idx+1)%test_shape[-1]==0:
        save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, lower_bounds, upper_bounds,save_bounds,original_shape=lambda x: original_shape(x[:idx+1],test_shape))

    elif (idx+1)%50==0 and idx<test_shape[-1]:
        save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, lower_bounds, upper_bounds,save_bounds,original_shape=lambda x:x[:idx+1])

save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, lower_bounds, upper_bounds, save_bounds,original_shape=lambda x: original_shape(x[:idx+1],test_shape))

del test_dataset
del test_ds
