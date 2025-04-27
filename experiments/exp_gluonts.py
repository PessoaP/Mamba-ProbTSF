# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os


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

import argparse

# Define parser
parser = argparse.ArgumentParser()

# Add arguments with default values
parser.add_argument('--folder', type=str, default='Sines_se_Linear_96_96_S_Mamba_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0')
parser.add_argument('--prediction_length', type=int, default=96)
parser.add_argument('--past_length', type=int, default=96)
parser.add_argument('--mamba_batch_size', type=int, default=16)
parser.add_argument('--perbatch', type=int, default=321)
parser.add_argument('--Method', type=str, default='Simple')

# Parse args
args = parser.parse_args()

# Assign variables
folder = 'results_prob/'+ args.folder
prediction_length = args.prediction_length
past_length = args.past_length
mamba_batch_size = args.mamba_batch_size
perbatch = args.perbatch
Method = args.Method

batch_size = mamba_batch_size * perbatch

# Now you can use them normally
print(f"Folder: {folder}")
print(f"Past length: {past_length}")
print(f"Prediction length: {prediction_length}")
print(f"Batch size: {batch_size}")
print(f"Method: {Method}")


# %%
#Loads same training and testing dataset as used in Mamba-ProbTSF, but in gluonts format
#This requires one to have run the Mamba-ProbTSF before to use the were data was retrained.

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

# %%
custom_dataset.shape,test_dataset.shape

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
        trainer=Trainer(ctx="cpu", epochs=40, learning_rate=1e-3),
    )

elif Method == 'DeepAR':
    from gluonts.mx import DeepAREstimator, Trainer
    estimator = DeepAREstimator(
        freq=freq,
        batch_size=batch_size,
        prediction_length=prediction_length,
        context_length=past_length,
        trainer=Trainer(ctx="cpu", epochs=40, learning_rate=1e-3),
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

# %%
#del train_ds
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds[:25],  # test dataset
    predictor=predictor,  # predictor
    num_samples=2000,  # number of sample paths we want for evaluation
)

# %%
forecasts = list(forecast_it)
tss = list(ts_it)


# %%
plt.plot(tss[-1].to_timestamp())
forecasts[-1].plot(show_label=True)
plt.legend()



# %%
from gluonts.evaluation import Evaluator

# %%
confidence = np.array([0.683, 0.955, 0.998])
quantiles = (1 + np.array([[-1], [1]]) * ( confidence)).T / 2

means = []
medians = []
futures = []
within_intervals = [[],[],[]]

for i in range(len(test_ds)):
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds[i:i+1],  # test dataset
        predictor=predictor,  # predictor
        num_samples=2000,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)[0]
    tss = list(ts_it)[0].to_numpy().reshape(-1)
    
    past,fut = tss[:-prediction_length],tss[-prediction_length:]

    means.append(forecasts.mean)
    medians.append(forecasts.median)
    futures.append(fut)

    for i in range(3):
        mino,majo = forecasts.quantile(quantiles[i,0]),forecasts.quantile(quantiles[i,1])
        within = np.logical_and(fut<majo,fut>mino)
        within_intervals[i].append(within.mean(axis=0))
    

# %%
import os
folder_path = folder + '/gluonts/'+ Method + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def original_shape(array,shape=test_shape):
    return array.reshape(-1,shape[2],array.shape[-1]).transpose(0,2,1)

np.save(folder_path + 'medians.npy',original_shape(np.array(medians)))
print('median avg error', (np.abs(np.array(medians)-np.array(futures))).mean(axis=0))

np.save(folder_path + 'means.npy',original_shape(np.array(means)))
print('mean avg error', (np.abs(np.array(means)-np.array(futures))).mean(axis=0))

[np.save(folder_path + 'sigma{}.npy'.format(i+1), np.array(within_intervals[i])) for i in range(3)]
print('within_intervals', [np.array(within_intervals[i]).mean() for i in range(3)])

# %%
print('Should be small:'(original_shape(np.array(futures))-fut[:2]).max())

# %%



