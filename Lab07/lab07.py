import pandas as pd
import pymc as pm
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("auto-mpg.csv")

data = data[data['horsepower'] != '?']
data['horsepower'] = data['horsepower'].astype(float)

sns.scatterplot(x='horsepower', y='mpg', data=data)
plt.title('Relația dintre Horsepower și Mile per gallon')
plt.xlabel('Horsepower')
plt.ylabel('Mile per gallon')
plt.show()


with pm.Model() as model:
    # variabilele
    beta0 = pm.Normal('beta0', mu=0, tau=1 / 10**2)
    beta1 = pm.Normal('beta1', mu=0, tau=1 / 10**2)

    # Modelul de regresie
    mu = beta0 + beta1 * data['horsepower']

    # Precizia erorii
    precision = pm.Gamma('precision', alpha=0.1, beta=0.1)

    # Distributia normala pentru variabila dependenta
    mpg_obs = pm.Normal('mpg_obs', mu=mu, tau=1 / precision, observed=data['mpg'])

# dreapta de regresie
with model:
    trace = pm.sample(2000, tune=1000)


pm.summary(trace).round(2)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='horsepower', y='mpg', data=data)

pm.plot_trace(trace)
plt.show()
