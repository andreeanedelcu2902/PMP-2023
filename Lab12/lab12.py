import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom

# Exercitiul 1

# variabila grid pentru metoda grid computing
grid = np.linspace(0, 1, 100)

#  estimam π folosind metoda grid computing
def estimate_pi_grid(N, prior):
    np.random.seed(42)
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    inside_circle = (x**2 + y**2 <= 1).astype(int)
    posterior = inside_circle * prior[:N]
    pi_estimate = np.sum(posterior) / N 

    return pi_estimate

# diferite distributii a priori pentru metoda grid computing
prior1 = (grid <= 0.5).astype(int)
prior2 = np.abs(grid - 0.5)
prior3 = np.random.rand(len(grid))

# estimam π cu diferite distributii a priori pentru metoda grid computing
pi_estimate1 = estimate_pi_grid(len(grid), prior1)
pi_estimate2 = estimate_pi_grid(len(grid), prior2)
pi_estimate3 = estimate_pi_grid(len(grid), prior3)

print("Estimarea lui π cu prior1:", pi_estimate1)
print("Estimarea lui π cu prior2:", pi_estimate2)
print("Estimarea lui π cu prior3:", pi_estimate3)

# Exercitiul 2

# diferite valori ale lui N
N_values = [100, 1000, 10000]
num_experiments = 100

mean_errors = []
std_errors = []

# estimarea lui π 
def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi_estimate = inside.sum() * 4 / N
    error = abs((pi_estimate - np.pi) / pi_estimate) * 100
    
    return error

# rulam codul de mai multe ori cu acelasi N si calculam eroarea
for N in N_values:
    errors = []

    for _ in range(num_experiments):
        error = estimate_pi(N)
        errors.append(error)

    mean_errors.append(np.mean(errors))
    std_errors.append(np.std(errors))

# rezultatele
plt.errorbar(N_values, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
plt.xlabel('Numarul de puncte (N)')
plt.ylabel('Eroare')
plt.title('Relatia dintre N si eroare in estimarea lui π')
plt.show()


# Exercitiul 3

def metropolis(func, draws=10000):
    """implementare metropolis simpla"""
    trace = np.zeros(draws)
    old_x = func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, func.std(), draws)
    
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
            
    return trace

# definim parametrii distributiei binomiale
n_params = [1, 2, 4]  # numarul de incercari
p_params = [0.25, 0.5, 0.75]  # probabilitatea de succes

# functia a priori ca o distributie beta
alpha = 2
beta_ = 5
prior_distribution = beta(alpha, beta_)

# alegem un set specific de parametri
n = 2
p = 0.5

# definim functia binomiala pentru acesti parametri
binomial_func = binom(n, p)

# apelam functia metropolis cu distributia a priori
samples = metropolis(prior_distribution, draws=10000)

# rezultatele
plt.hist(samples, bins=30, density=True, alpha=0.5, label='Metropolis')
plt.plot(np.linspace(0, 1, 100), prior_distribution.pdf(np.linspace(0, 1, 100)), 'r-', label='Distributia Prior')
plt.plot(np.arange(0, n+1), binomial_func.pmf(np.arange(0, n+1)), 'g-', label='Distributia binomiala')
plt.xlabel('Valoarea parametrului')
plt.ylabel('Densitatea de probabilitate')
plt.legend()
plt.show()
