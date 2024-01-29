import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


#EX 1

# a.  pandas DataFrame
df = pd.read_csv("BostonHousing.csv")

#  coloanele dorite
data = df[["medv", "rm", "crim", "indus"]]

# b. estimarile de 95%
with pm.Model() as model:
    # priors pentru coeficienti
    beta_rm = pm.Normal("beta_rm", mu=0, sd=1)
    beta_crim = pm.Normal("beta_crim", mu=0, sd=1)
    beta_indus = pm.Normal("beta_indus", mu=0, sd=1)
    alpha = pm.Normal("alpha", mu=0, sd=1)

    # model liniar
    mu = alpha + beta_rm * data["rm"] + beta_crim * data["crim"] + beta_indus * data["indus"]

    # 
    sigma = pm.HalfCauchy("sigma", beta=1)
    medv = pm.Normal("medv", mu=mu, sd=sigma, observed=data["medv"])

    # sampling
    trace = pm.sample(2000, tune=1000, cores=2)

# sumarul modelului
summary = az.summary(trace, hdi_prob=0.95)
print(summary)

# c. extragerile
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=2000)

#  intervalul de predictie de 50% HDI pentru valoarea locuintelor
hdi_50 = az.hdi(post_pred["medv"], hdi_prob=0.5)

print("Interval de predictie 50% HDI pentru valoarea locuintelor:")
print(hdi_50)



#EX 2

def posterior_grid(grid_points=50, data=None):
    """
    A grid implementation for the coin-flipping problem with geometric distribution
    """
    grid = np.linspace(0, 1, grid_points)
    prior = np.ones_like(grid)  # Uniform prior
    likelihood = stats.geom.pmf(data, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

# aruncam moneda de la infinit si vedem prima stema la a 5-a aruncare
data = 5
points = 50  

# distributia a posteriori
grid, posterior = posterior_grid(points, data)


plt.plot(grid, posterior, 'o-')
plt.title(f'Observație: prima stema la a 5-a aruncare')
plt.yticks([])
plt.xlabel('θ')
plt.show()

# valoarea theta care maximizeaza probabilitatea a posteriori
theta_maxim = grid[np.argmax(posterior)]
print(f'Valoarea theta care maximizează probabilitatea a posteriori: {theta_maxim}')
