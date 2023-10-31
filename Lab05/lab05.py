import pymc3 as pm
import numpy as np
import pandas as pd


data = pd.read_csv('trafic.csv')


ore_crestere = [7, 16]
ore_descrestere = [8, 19]


with pm.Model() as model:
  
    lambda_ = pm.Uniform('lambda', 0, 10)  # Intervalul poate fi ajustat în funcție de așteptări

    traffic = pm.Poisson('traffic', lambda_, observed=data['traffic'])

    lambda_crestere = pm.Normal('lambda_crestere', mu=lambda_, sd=1, shape=len(ore_crestere))
    lambda_descrestere = pm.Normal('lambda_descrestere', mu=lambda_, sd=1, shape=len(ore_descrestere))


    trafic_modificat = pm.math.zeros(len(data))
    for i, minute in enumerate(range(len(data))):
        if minute // 60 in ore_crestere:
            trafic_modificat = pm.math.set_subtensor(trafic_modificat[i], lambda_crestere[ore_crestere.index(minute // 60)])
        elif minute // 60 in ore_descrestere:
            trafic_modificat = pm.math.set_subtensor(trafic_modificat[i], lambda_descrestere[ore_descrestere.index(minute // 60)])


    lambda_final = lambda_ + trafic_modificat


    traffic_final = pm.Poisson('traffic_final', lambda_final, observed=data['traffic'])


with model:
    trace = pm.sample(1000, tune=1000, cores=4)

pm.summary(trace)

