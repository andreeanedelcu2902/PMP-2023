from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Defining the model structure.
model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Cutremur', 'Alarma'), ('Incendiu', 'Alarma')])

# Defining individual CPDs 
cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])
cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, values=[[0.01, 0.03], [0.99, 0.97]], evidence=['Cutremur'], evidence_card=[2])
cpd_alarma = TabularCPD(variable='Alarma', variable_card=2, values=[[0.98, 0.02, 0.03, 0.97], [0.02, 0.98, 0.97, 0.03]], evidence=['Cutremur', 'Incendiu'], evidence_card=[2, 2])

# Associating the CPDs with the network
model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarma)

# Verifying the model
assert model.check_model()

# 2. Calculați probabilitatea ca un cutremur să aibă loc, dată fiind activarea alarmei de incendiu
inference = VariableElimination(model)
result = inference.query(variables=['Cutremur'], evidence={'Alarma': 1})
prob_cutremur_dupa_alarma = result.values[1]
print("Probabilitatea ca un cutremur să aibă loc după activarea alarmei de incendiu:", prob_cutremur_dupa_alarma)

# 3. Afișați probabilitatea ca un incendiu să aibă loc fără activarea alarmei de incendiu
result = inference.query(variables=['Incendiu'], evidence={'Alarma': 0})
prob_incendiu_fara_alarma = result.values[1]
print("Probabilitatea ca un incendiu să aibă loc fără activarea alarmei de incendiu:", prob_incendiu_fara_alarma)

# Performing exact inference using Variable Elimination
inference = VariableElimination(model)
result = inference.query(variables=['Cutremur'], evidence={'Alarma': 1})
prob_cutremur_dupa_alarma = result.values[1]
print("Probabilitatea ca un cutremur să aibă loc după activarea alarmei de incendiu:", prob_cutremur_dupa_alarma)



pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()