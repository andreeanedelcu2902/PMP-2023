import pymc3 as pm
import arviz as az
import numpy as np

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

# Definirea modelului Bayesian
with pm.Model() as model:
    n = pm.Poisson("n", mu=10)
    
    Y_obs = pm.Binomial("Y_obs", n=n, p=theta_values, observed=Y_values)

    # Sampling
    trace = pm.sample(2000, tune=1000, target_accept=0.9)


az.plot_posterior(trace)

