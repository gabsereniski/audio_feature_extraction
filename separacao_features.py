import numpy as np
from sklearn.model_selection import train_test_split

# Carregar os dados a partir do arquivo txt
data = np.genfromtxt('features.txt', dtype=str)

# Separar as features dos rótulos
features = data[:, :-1]
labels = data[:, -1]

# Dividir os dados em treinamento e teste, mantendo a proporção das classes
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, stratify=labels)

# Salvar os dados de treinamento em um arquivo
train_data = np.column_stack((X_train, y_train))
np.savetxt('train_data.txt', train_data, fmt='%s')

# Salvar os dados de teste em um arquivo
test_data = np.column_stack((X_test, y_test))
np.savetxt('test_data.txt', test_data, fmt='%s')

# Exibir o tamanho dos conjuntos de treinamento e teste
print("Tamanho do conjunto de treinamento:", X_train.shape[0])
print("Tamanho do conjunto de teste:", X_test.shape[0])
