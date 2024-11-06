import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from supervisedLearning import supervised_learning
from unsupervisedLearning import kmeans_clustering
from semanticIntegration import get_breast_cancer_knowledge, enrich_data_with_knowledge
from bayesianNetwork import create_bayesian_network


def preprocess_data(data):

    if 'id' in data.columns:
        data = data.drop('id', axis=1)
    

    if 'target' in data.columns:
        data['target'] = data['target'].map({'M': 1, 'B': 0})

    data = data.dropna(axis=1, how='all')

    return data

dataset_path = "data.csv"
data = pd.read_csv(dataset_path)
data = preprocess_data(data)
data.to_csv('dataProcessato.csv', index=False)

#controllo sui valori nulli
print(data.isnull().sum())

target_column = 'target'

print("\n\nSistema di diagnosi del cancro al seno\n")
print()

X = data.drop(target_column, axis=1)
y = data[target_column]

# distribuzione delle diagnosi
benigno = data.target.value_counts()[0]
maligno = data.target.value_counts()[1]

print()
print('Pazienti con cancro benigno:', benigno, '(% {:.2f})'.format(benigno / data.target.count() * 100))
print('Pazienti con cancro maligno:', maligno, '(% {:.2f})'.format(maligno / data.target.count() * 100), '\n')

# grafico della distribuzione
labels = ['Benigno', 'Maligno']
sizes = [benigno, maligno]
colors = ['skyblue', 'lightcoral']
explode = (0.1, 0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribuzione delle diagnosi')
plt.show()

# Integrazione con conoscenza semantica
knowledge = get_breast_cancer_knowledge()

X_enriched = enrich_data_with_knowledge(X, knowledge)
'''print(X_enriched.isnull().sum()) '''
X_enriched.to_csv('semantica.csv', index=False)

X_selected, y = supervised_learning(X_enriched, y)
'''print(X_selected.isnull().sum())
X_selected.to_csv('supervisionato.csv', index=False)'''

X_clustered = kmeans_clustering(X_selected)
#X_clustered_em = perform_em_clustering(X_selected)

'''print(X_clustered.isnull().sum())
X_clustered.to_csv('K_non_sup.csv', index=False)'''

create_bayesian_network(data, X_selected, y, target_column)
