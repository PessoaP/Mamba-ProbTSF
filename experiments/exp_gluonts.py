# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import os
from tqdm import tqdm
import argparse


from gluonts.dataset.repository import get_dataset, dataset_names
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions

# %%
# folder = 'results_prob/Sines_se_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0'
# prediction_length, past_length = 96,96 #Brownian is different
# mamba_batch_size, perbatch = 16,321  #Traffic is different
# batch_size = mamba_batch_size*perbatch

# Method = 'Simple'
# Method = 'DeepAR'
# Method = 'GP'
# script.py


# Define parser
parser = argparse.ArgumentParser()

# Add arguments with default values
parser.add_argument('--folder', type=str, default='ECL_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0')#'Sines_se_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0')
parser.add_argument('--prediction_length', type=int, default=96)
parser.add_argument('--past_length', type=int, default=96)
parser.add_argument('--mamba_batch_size', type=int, default=16)
parser.add_argument('--Method', type=str, default='Simple')
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



past = np.load(folder+'/train_dataset/train_input.npy')
fut  = np.load(folder+'/train_dataset/train_trues.npy')

past = np.transpose(past, (0, 2, 1)).reshape(-1,past_length)
fut  = np.transpose(fut , (0, 2, 1)).reshape(-1,prediction_length)

custom_dataset = np.hstack((past,fut))

print(test_dataset.shape,custom_dataset.shape)

# %%

def original_shape(array, shape=test_shape):
    reshaped = array.reshape(-1, shape[2], array.shape[-1])
    return reshaped.transpose(0, 2, 1)

def save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, 
                           lower_bounds=None, upper_bounds=None, save_bounds=save_bounds,original_shape=original_shape):
    os.makedirs(folder_path, exist_ok=True)

    medians_array = np.array(medians)
    np.save(os.path.join(folder_path, 'medians.npy'), original_shape(medians_array))
    print('Median avg error:', np.abs(medians_array - np.array(futures)).mean(axis=0))

    means_array = np.array(means)
    np.save(os.path.join(folder_path, 'means.npy'), original_shape(means_array))
    print('Mean avg error:', np.abs(means_array - np.array(futures)).mean(axis=0))

    for k in range(3):
        np.save(os.path.join(folder_path, f'sigma{k+1}.npy'), original_shape(np.array(within_intervals[k])))
        print(f'Within {confidence[k]*100:.1f}% interval:', np.array(within_intervals[k]).mean())

        if save_bounds and lower_bounds is not None and upper_bounds is not None:
            np.save(os.path.join(folder_path, f'sigma{k+1}_lower.npy'), original_shape(np.array(lower_bounds[k])))
            np.save(os.path.join(folder_path, f'sigma{k+1}_upper.npy'), original_shape(np.array(upper_bounds[k])))

def process_single_forecast(test_sample, predictor, prediction_length, quantiles, save_bounds=save_bounds):
    """
    Process a single time series forecast and return forecast stats.
    """
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_sample,
        predictor=predictor,
        num_samples=1000,
    )
    forecast = list(forecast_it)[0]
    ts = list(ts_it)[0].to_numpy().reshape(-1)

    past, fut = ts[:-prediction_length], ts[-prediction_length:]

    forecast_means = forecast.mean
    forecast_medians = forecast.median
    forecast_futures = fut

    forecast_within_intervals = []
    forecast_lower_bounds = []
    forecast_upper_bounds = []

    for k in range(3):
        lower = forecast.quantile(quantiles[k, 0])
        upper = forecast.quantile(quantiles[k, 1])
        within = np.logical_and(fut > lower, fut < upper)
        forecast_within_intervals.append(within)

        if save_bounds:
            forecast_lower_bounds.append(lower)
            forecast_upper_bounds.append(upper)

    return (forecast_means, forecast_medians, forecast_futures, 
            forecast_within_intervals, forecast_lower_bounds, forecast_upper_bounds)



perbatch = test_shape[-1]
batch_size = mamba_batch_size * perbatch
print(f"Batch size: {batch_size}")
# %%
freq = "1H"
start = pd.Period("01-01-2019", freq=freq)  # Just part of gluonts test, nothing to be attributed from this



# %%
# train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
train_ds = ListDataset(
    [{"target": x, "start": start} for x in custom_dataset],
    freq=freq,
)

# test dataset: use the whole dataset, add "target" and "start" fields
test_ds = ListDataset(
    [{"target": x, "start": start} for x in test_dataset], freq=freq
)



# %%
if Method == 'FeedForward':
    from gluonts.mx import SimpleFeedForwardEstimator, Trainer
    estimator = SimpleFeedForwardEstimator(
        batch_size=batch_size,
        num_hidden_dimensions=[10],
        prediction_length=prediction_length,
        context_length=past_length,
        trainer=Trainer(ctx="cpu", epochs=20, learning_rate=1e-3),
    )

elif Method == 'DeepAR':
    from gluonts.mx import DeepAREstimator, Trainer
    estimator = DeepAREstimator(
        freq=freq,
        batch_size=batch_size,
        prediction_length=prediction_length,
        context_length=past_length,
        trainer=Trainer(ctx="cpu", epochs=20, learning_rate=1e-3),
    )

elif Method == 'GP':
    from gluonts.mx import GaussianProcessEstimator, Trainer
    estimator =GaussianProcessEstimator(
        freq=freq,
        cardinality=1,
        batch_size=batch_size,
        prediction_length=prediction_length,
        context_length=past_length,
        trainer=Trainer(ctx="cpu", epochs=40, learning_rate=1e-3),
    )


# %%
predictor = estimator.train(train_ds)

# %%Training is done

del custom_dataset
del train_ds
# --- Helper functions ---



# --- Configuration ---
confidence = np.array([0.683, 0.955, 0.998])
quantiles = (1 + np.array([[-1], [1]]) * confidence).T / 2  # shape (3,2)

# Output folder
folder_path = os.path.join(folder, 'gluonts', Method)
os.makedirs(folder_path, exist_ok=True)


# --- Storage ---
means = []
medians = []
futures = []

within_intervals = [[] for _ in range(3)]
lower_bounds = [[] for _ in range(3)] if save_bounds else None
upper_bounds = [[] for _ in range(3)] if save_bounds else None


# --- Main loop ---
means = []
medians = []
futures = []
within_intervals = [[] for _ in range(3)]
lower_bounds = [[] for _ in range(3)] if save_bounds else None
upper_bounds = [[] for _ in range(3)] if save_bounds else None

for idx in tqdm(range(batch_size//4)):
    (mean_val, median_val, future_val, within_vals, lower_vals, upper_vals) = process_single_forecast(
        test_ds[idx:idx+1], predictor, prediction_length, quantiles, save_bounds=save_bounds
    )

    means.append(mean_val)
    medians.append(median_val)
    futures.append(future_val)

    for k in range(3):
        within_intervals[k].append(within_vals[k])

        if save_bounds:
            lower_bounds[k].append(lower_vals[k])
            upper_bounds[k].append(upper_vals[k])

    if (idx+1)%test_shape[-1]==0:
        save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, lower_bounds, upper_bounds,save_bounds)
    elif (idx+1)%50==0 and idx<test_shape[-1]:
        save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, lower_bounds, upper_bounds,save_bounds,original_shape=lambda x:x)

save_forecast_outputs(folder_path, medians, means, futures, within_intervals, confidence, lower_bounds, upper_bounds, save_bounds)


