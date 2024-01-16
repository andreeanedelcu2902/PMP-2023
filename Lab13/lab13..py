import arviz as az
import matplotlib.pyplot as plt

#  datele pentru cele doua modele
idata_cm = az.load_arviz_data("centered_eight")
idata_ncm = az.load_arviz_data("non_centered_eight")

# 1.distributia a posteriori pentru fiecare model
#nr de lanturi si marimea totala a esantionului
for idx, tr in enumerate([idata_cm, idata_ncm]):
    print(f"Model {'centered' if idx == 0 else 'non-centered'}:")
    print(f"Numarul de lanturi: {tr.posterior.chain.size}")
    print(f"Marimea totala a esantionului: {tr.posterior.chain.size * tr.posterior.draw.size}\n")

    # distributia a posteriori
    az.plot_posterior(tr)
    plt.title(['centered', 'non-centered'][idx])
    plt.show()
    
#2
# Rhat pentru parametrii 'mu' și 'tau'
rhat_cm_mu = az.rhat(idata_cm.posterior['mu'])
rhat_cm_tau = az.rhat(idata_cm.posterior['tau'])
rhat_ncm_mu = az.rhat(idata_ncm.posterior['mu'])
rhat_ncm_tau = az.rhat(idata_ncm.posterior['tau'])

# afisam
print(f"Rhat pentru parametrul 'mu' în modelul centrat: {rhat_cm_mu}")
print(f"Rhat pentru parametrul 'tau' în modelul centrat: {rhat_cm_tau}")
print(f"Rhat pentru parametrul 'mu' în modelul necentrat: {rhat_ncm_mu}")
print(f"Rhat pentru parametrul 'tau' în modelul necentrat: {rhat_ncm_tau}")

#  autocorelație pentru parametrii 'mu' și 'tau'
az.plot_autocorr(idata_cm, var_names=['mu', 'tau'], figsize=(10, 5))
plt.title('Autocorelație în modelul centrat')
plt.show()

az.plot_autocorr(idata_ncm, var_names=['mu', 'tau'], figsize=(10, 5))
plt.title('Autocorelație în modelul necentrat')
plt.show()


# 3. nr de divergente si vizualizarea in spatiul parametrilor (mu si tau)
for idx, tr in enumerate([idata_cm, idata_ncm]):
    divergences_count = tr.sample_stats.diverging.sum()
    print(f"Numarul de divergente pentru {'centered' if idx == 0 else 'non-centered'} model: {divergences_count}")

    # afisam divergentele in spatiul parametrilor
    _, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5), constrained_layout=True)
    az.plot_pair(tr, var_names=['mu', 'tau'], kind='scatter', divergences=True, divergences_kwargs={'color':'C1'}, ax=ax)
    ax[0].set_title(['centered', 'non-centered'][idx])
    plt.show()
