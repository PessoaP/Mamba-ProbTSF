# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# %%
np.random.seed(42)
chain_size= 25000
chains = 321

#tempora 1 and 0
diff_coeff = 1+0*np.exp(np.random.normal(size=(1,chains)))
drift_coeff = 0*np.tan((np.pi/2)*(np.random.uniform(size=(1,chains))-1/2))

# %%
eta = np.random.normal(size=(chains,chain_size))
noise = diff_coeff*eta.cumsum(axis=1).T
drift =  drift_coeff*(np.arange(chain_size).reshape(-1,1))
traces = drift+noise

# %%
df = pd.DataFrame(traces.astype(np.float32),columns=1+np.arange(chains))

df.index = np.arange(chain_size)
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'date'})
df.set_index('date', inplace=True)

# %%
df.to_csv('brownian.csv')

# %%



