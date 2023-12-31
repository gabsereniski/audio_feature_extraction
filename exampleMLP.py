#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo do Algoritmo k-NN para scikit-learn.
@author: diegobertolini
"""
import sys
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.stats import mode

def main():
    # Carregar dados de treino
    train_data = pd.read_csv("train_data.txt", delimiter=' ', header=None)
    X_train = train_data.iloc[:, :-1].values  # Atributos
    labels_train = train_data.iloc[:, -1].values  # Rótulos

    # Codificar rótulos de treino
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(labels_train)

    # Carregar dados de teste
    test_data = pd.read_csv("test_data.txt", delimiter=' ', header=None)
    X_test = test_data.iloc[:, :-1].values  # Atributos
    labels_test = test_data.iloc[:, -1].values  # Rótulos

    # Codificar rótulos de teste
    y_test = label_encoder.transform(labels_test)

    ## k-NN classifier
    neigh = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    neigh.fit(X_train, y_train)
    y_pred_knn = neigh.predict(X_test)
    print("KNN")
    print(classification_report(y_test, y_pred_knn, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_knn))
   
###SVM com Grid search
    C_range = 2.0 ** np.arange(-5, 15, 2)
    gamma_range = 2.0 ** np.arange(3, -15, -2)
    
    k = ["rbf"]
    # instancia o classificador, gerando probabilidades
    srv = svm.SVC(probability=True, kernel="rbf")
    ss = StandardScaler()
    pipeline = Pipeline([("scaler", ss), ("svm", srv)])
    
    param_grid = {"svm__C": C_range, "svm__gamma": gamma_range}

    # faz a busca
    grid = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=True)
    grid.fit(X_train, y_train)

    # recupera o melhor modelo
    model = grid.best_estimator_
    y_pred_svm = model.predict(X_test)
    print("SVM")
    print(classification_report(y_test, y_pred_svm, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))

### MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = MLPClassifier(
        solver="lbfgs",
        alpha=1e-5,
        hidden_layer_sizes=(500, 500, 500, 500),
        random_state=1,
        max_iter=10000
    )
    clf.fit(X_train, y_train)
    print("MLP, lbfgs")
    print(classification_report(y_test, clf.predict(X_test), digits=4))
    
    y_pred_mlp_lbfgs = clf.predict(X_test_scaled)
    print("MLP, lbfgs")
    print(classification_report(y_test, y_pred_mlp_lbfgs, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_mlp_lbfgs))
    
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    clf = MLPClassifier(
        solver="sgd",
        alpha=1e-5,
        hidden_layer_sizes=(500, 500, 500, 500),
        random_state=1,
        max_iter=10000
    )
    clf.fit(X_train, y_train)
    y_pred_mlp_sgd = clf.predict(X_test_scaled)
    print("MLP, sgd")
    print(classification_report(y_test, y_pred_mlp_sgd, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_mlp_sgd))
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    clf = MLPClassifier(
        solver="adam",
        alpha=1e-5,
        hidden_layer_sizes=(500, 500, 500, 500),
        random_state=1,
        max_iter=1000
    )
    clf.fit(X_train, y_train)
    y_pred_mlp_adam = clf.predict(X_test_scaled)
    print("MLP, adam")
    print(classification_report(y_test, y_pred_mlp_adam, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_mlp_adam))


### Random Forest Classifier
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    clf = RandomForestClassifier(n_estimators=10000, max_depth=30, random_state=1)
    clf.fit(X_train, y_train)
    y_pred_rf = clf.predict(X_test)
    print("Random Forest")
    print(classification_report(y_test, y_pred_rf, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
   
### Decision Tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred_dt = clf.predict(X_test)
    print("Decision Tree")
    print(classification_report(y_test, y_pred_dt, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_dt))
    tree.plot_tree(clf)
# Combinação dos resultados
    y_pred_combined = mode(
        np.array([
            y_pred_knn,
            y_pred_svm,
            y_pred_mlp_lbfgs,
            y_pred_mlp_sgd,
            y_pred_mlp_adam,
            y_pred_rf,
            y_pred_dt
        ]),
        axis=0
    )[0][0]

    # Métricas de desempenho para a combinação dos classificadores
    print("Combined Results")
    print(classification_report(y_test, y_pred_combined, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_combined))
if __name__ == "__main__":
    main()
