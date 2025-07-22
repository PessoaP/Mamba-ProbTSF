

# **Mamba-ProbTSF: Mamba Time Series Forecasting with Uncertainty Quantification**  

This repository implements a dual-network framework based on the Mamba architecture for probabilistic time series forecasting. One network produces point forecasts, while the other estimates point-wise uncertainty for the forecasted region.  

We discuss the motivation and impact of this approach in our manuscript:  
"Mamba Time Series Forecasting with Uncertainty Quantification", published on IOP's [Machine learning: science and technology](https://doi.org/10.1088/2632-2153/adec3b) and as preprint on [arXiv](https://arxiv.org/abs/2503.10873)

---
<div style="background-color:white; display:inline-block; padding:10px;">
  <img src="https://arxiv.org/html/2503.10873v1/x1.png" width="750"/>
</div>

## **How to run the code**  
To reproduce the analysis/figures from our manuscript, follow these steps:  

1. **Generate synthetic data** by running:  
   ```bash
   bash synthetic_data.sh

2. **Obtaining the real datasets** from [here](https://github.com/wzhwzhwzh0921/S-D-Mamba)

3. **Run forecasting scripts for different datasets**  
Use the following scripts to run experiments on different datasets:  
- **Sines dataset:**  
  ```bash
  bash sines_script.sh
- **Van der pol dataset:**  
  ```bash
  bash VDP_script.sh
- **Electricity consumption dataset dataset:**  
  ```bash
  bash ECL_script.sh
- **Traffic occupancy dataset:**
  ```bash
  bash traffic_script.sh
- **Brownian motion dataset:**  
  ```bash
  brownian_script.sh

4. **Generate the figures** using the iPython notebook `make_figures.ipynb`

## **Citation**
```
@article{pessoa2025mamba,
  title   = {Mamba time series forecasting with uncertainty quantification},
  author  = {Pedro Pessoa and Paul Campitelli and Douglas P. Shepherd and S. Banu Ozkan and Steve Pressé},
  journal = {Machine Learning: Science and Technology},
  volume  = {6},
  pages   = {035012},
  year    = {2025},
  doi     = {10.1088/2632-2153/adec3b}
}

```

## **Acknowledgments**
Initial versions of this code were build on top of S-Mamba (https://github.com/wzhwzhwzh0921/S-D-Mamba accompanied with paper https://www.sciencedirect.com/science/article/pii/S0925231224019490). With the express authorization of the authors, we have adapted and modified the original implementation.

