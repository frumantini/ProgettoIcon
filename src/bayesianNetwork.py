import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import K2Score, HillClimbSearch, BayesianEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

def create_bayesian_network(data, X, y, target_column):
    print("\nCreazione e analisi della Rete Bayesiana:")
    
    k_best = min(10, X.shape[1])
    selector = SelectKBest(f_classif, k=k_best)
    X_reduced = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()

    X_selected = pd.DataFrame(X_reduced, columns=selected_features)

    n_bins = 3
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_discretized = discretizer.fit_transform(X_selected)

    data_processed = pd.DataFrame(X_discretized, columns=selected_features)
    data_processed[target_column] = y.values

    print("Creazione della struttura della rete bayesiana...")
    k2 = K2Score(data_processed)
    hc_k2 = HillClimbSearch(data_processed)

    k2_model = hc_k2.estimate(scoring_method=k2, max_indegree=3, max_iter=int(1e4))

    print("Creazione e addestramento della rete bayesiana...")
    bNet = BayesianNetwork(k2_model.edges())
    bNet.fit(data_processed, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)

    print("Nodi della rete bayesiana:", bNet.nodes())
    print("Archi della rete bayesiana:", bNet.edges())

    plot_bayesian_network(bNet)

    data_inference = VariableElimination(bNet)

    perform_complex_inference(data_inference, bNet, selected_features, target_column, discretizer)

def plot_bayesian_network(bNet):
    G = nx.DiGraph()
    G.add_edges_from(bNet.edges())
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue', alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
    plt.title("Rete Bayesiana per la Diagnosi del Cancro al Seno", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def perform_complex_inference(data_inference, bNet, selected_features, target_column, discretizer):
    print("\nEsecuzione di inferenze complesse:")

    # Scenario 1: Probabilità di cancro maligno dato un insieme di caratteristiche del tumore
    evidence_malignant = {
        'radius_mean': 2,  # Valore alto
        'perimeter_mean': 2,  
        'area_mean': 2,  
    }
    prob_malignant = data_inference.query(variables=[target_column], evidence=evidence_malignant)
    print(f"\nScenario 1a - Probabilità di cancro maligno date le caratteristiche {evidence_malignant}:")
    print(prob_malignant)

    # Scenario 1b: Probabilità di cancro benigno dato un insieme di caratteristiche del tumore
    evidence_benign = {
        'radius_mean': 0,  # Valore basso
        'perimeter_mean': 0,  
        'area_mean': 0,  
    }
    prob_benign = data_inference.query(variables=[target_column], evidence=evidence_benign)
    print(f"\nScenario 1b - Probabilità di cancro benigno date le caratteristiche {evidence_benign}:")
    print(prob_benign)

    # Scenario 2: Identificazione delle caratteristiche più influenti
    print("\nScenario 2 - Analisi dell'influenza delle caratteristiche:")
    influence_analysis = {}
    for feature in selected_features:
        high_evidence = {feature: 2}  # Valore alto per la caratteristica
        prob_high = data_inference.query(variables=[target_column], evidence=high_evidence)
        low_evidence = {feature: 0}  # Valore basso per la caratteristica
        prob_low = data_inference.query(variables=[target_column], evidence=low_evidence)
        influence = prob_high.values[1] - prob_low.values[1]  # Differenza nella probabilità di cancro maligno
        influence_analysis[feature] = influence

    for feature, influence in sorted(influence_analysis.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{feature}: {influence:.4f}")

    # Scenario 3: Analisi di sensibilità
    print("\nScenario 3 - Analisi di sensibilità:")
    base_prob = data_inference.query(variables=[target_column]).values[1]
    for feature in selected_features:
        sensitivities = []
        for value in range(3):  # Considerando 3 bin di discretizzazione
            evidence = {feature: value}
            prob = data_inference.query(variables=[target_column], evidence=evidence).values[1]
            sensitivity = (prob - base_prob) / base_prob
            sensitivities.append(sensitivity)
        print(f"{feature}: Min={min(sensitivities):.4f}, Max={max(sensitivities):.4f}")

    # Scenario 4: Inferenza su caratteristiche nascoste
    print("\nScenario 4 - Inferenza su caratteristiche nascoste:")
    hidden_feature = 'concavity_mean'  # Esempio di caratteristica nascosta
    evidence = {
        'radius_mean': 2,
        'perimeter_mean': 2,
        'area_mean': 2
    }
    inferred_prob = data_inference.query(variables=[hidden_feature], evidence=evidence)
    print(f"Probabilità inferita per {hidden_feature} data l'evidenza {evidence}:")
    print(inferred_prob)

def create_evidence(values, selected_features, discretizer):
    selected_values = values[selected_features]
    discretized_values = discretizer.transform([selected_values])[0]
    return {feature: int(value) for feature, value in zip(selected_features, discretized_values)}