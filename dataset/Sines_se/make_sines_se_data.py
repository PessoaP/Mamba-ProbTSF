# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# %%
np.random.seed(42)
chain_size= 25000
chains = 321


# %%
xi = np.random.normal(size=(chain_size,chains))
phi1 = 2*np.pi*np.random.rand(chains)
t = np.arange(chain_size).reshape(-1,1)
w1 = 2*np.pi/24
w2 = w1 * np.random.exponential(size=chains)/2

# %%
traces = 4*np.sin(phi1+w1*t) + np.sin(w2*t) + xi

# %%
df = pd.DataFrame(traces.astype(np.float32),columns=1+np.arange(chains))

df.index = np.arange(chain_size)
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'date'})
df.set_index('date', inplace=True)

# %%
df.to_csv('sines.csv')

# %%



