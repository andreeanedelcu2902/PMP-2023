import random

def arunca_moneda_masluita():
    #  1 cu probabilitatea 2/3 și 0 cu probabilitatea 1/3
    return 1 if random.random() < 2/3 else 0

def arunca_moneda_normala():
    #  0 sau 1 cu probabilitatea 1/2
    return random.choice([0, 1])

def simuleaza_joc():
    castigator_runda_1 = arunca_moneda_normala()
    castigator_runda_2 = arunca_moneda_masluita()

    return castigator_runda_1 if castigator_runda_1 >= castigator_runda_2 else 1

def simulare_multipla(numar_jocuri):
    castiguri_j0 = 0
    castiguri_j1 = 0

    for _ in range(numar_jocuri):
        castigator = simuleaza_joc()
        if castigator == 0:
            castiguri_j0 += 1
        else:
            castiguri_j1 += 1

    procentaj_j0 = (castiguri_j0 / numar_jocuri) * 100
    procentaj_j1 = (castiguri_j1 / numar_jocuri) * 100

    return procentaj_j0, procentaj_j1

# simulam 10.000 de jocuri
rezultate = simulare_multipla(10000)
print(f"Procentaj de castig pentru j0: {rezultate[0]:.2f}%")
print(f"Procentaj de castig pentru j1: {rezultate[1]:.2f}%")
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

# reteaua bayesiana:
model = BayesianModel([('J0_moneda', 'J0_castig'), ('J1_moneda', 'J1_castig'), ('J0_castig', 'Castigator')])

CPD_j0 = TabularCPD(variable='JO_moneda', variable_card=2, values=[[0.5], [0.5]])
print(CPD_j0)
CPD_j1 = TabularCPD(variable='J1_moneda', variable_card=2, values=[[0.6], [0.4]])
print(CPD_j1)
cpd_c = TabularCPD(variable='J0_castig', variable_card=2, 
                   values=[[0.5, 0.5, 0.5, 0.0], 
                           [0.6, 0.5, 0.4, 1.0]],
                  evidence=['J0_moneda', 'Castigator'],
                  evidence_card=[2, 2])

cpd_d = TabularCPD(variable='J1_castig', variable_card=2, 
                   values=[[0.5, 0.5, 0.5, 0.0], 
                           [0.6, 0.5, 0.4, 1.0]],
                  evidence=['J1_moneda', 'Castigator'],
                  evidence_card=[2, 2])

# asociem cpd cu reteaua
model.add_cpds(CPD_j0, CPD_j1, cpd_c, cpd_d)

#verificam modelul
assert model.check_model()

#probabilitati conditionate
#model.fit(data, estimator=MaximumLikelihoodEstimator)


# calculam probabilitatile conditionate
inference = VariableElimination(model)
prob_castig_j0 = inference.query(variables=['Castigator'], evidence={'J0_moneda': 1, 'J1_moneda': 0}).values[0]
prob_castig_j1 = inference.query(variables=['Castigator'], evidence={'J0_moneda': 0, 'J1_moneda': 1}).values[0]

#afisam probabilitatile de castig 
print(f"Probabilitatea ca J0 sa castige: {prob_castig_j0:.2f}")
print(f"Probabilitatea ca J1 să castige: {prob_castig_j1:.2f}")

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()