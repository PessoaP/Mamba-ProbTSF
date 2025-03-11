# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# %%
np.random.seed(42)
chain_size= 25000
chains = 321


# %%
def differential(x,dxdt,mu=5.,omega = 2*np.pi/24):
    d2xdt2 = (mu*(1-x**2)*dxdt - x)*(omega**2)
    return dxdt,d2xdt2

def evolve(x, dxdt, dt,mu=5.,omega = 2*np.pi/24):
    # Compute the four slopes for RK4
    k1_dx, k1_d2x = differential(x, dxdt, mu, omega)
    k2_dx, k2_d2x = differential(x + k1_dx * dt / 2, dxdt + k1_d2x * dt / 2, mu, omega)
    k3_dx, k3_d2x = differential(x + k2_dx * dt / 2, dxdt + k2_d2x * dt / 2, mu, omega)
    k4_dx, k4_d2x = differential(x + k3_dx * dt, dxdt + k3_d2x * dt, mu, omega)
    
    # Update x and dxdt using weighted average of slopes
    x_f = x + (dt / 6) * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx)
    dxdt_f = dxdt + (dt / 6) * (k1_d2x + 2 * k2_d2x + 2 * k3_d2x + k4_d2x)
    
    return x_f, dxdt_f

def trajs(x0=np.zeros(chains), 
          v0=np.ones(chains), 
          mu=10*np.random.exponential(size=chains), 
          omega=2*np.pi/24,
          dt=1e-3, 
          T=chain_size, 
          burn=5):

    xs = []
    vs = []

    xt, vt = x0, v0
    t = 0
    while t < T+burn:
        for i in np.arange(1/dt):
            xt, vt = evolve(xt, vt, dt, mu, omega)
        t += 1
        if t>burn:
            xs.append(xt.copy())
            vs.append(vt.copy())

    return np.arange(len(xs)), np.array(xs), np.array(vs)



# %%
t,xs,vs = trajs()
traces = 4*xs + np.random.normal(size=(chain_size,chains))

# %%
plt.plot(4*xs[-96:,-1])
plt.plot(traces[-96:,-1])

# %%
df = pd.DataFrame(traces.astype(np.float32),columns=1+np.arange(chains))

df.index = np.arange(chain_size)
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'date'})
df.set_index('date', inplace=True)
df.to_csv('VdP.csv')

# %%



