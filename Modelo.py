import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.metrics

def graficador():
    for i in range(len(ii)):
        plt.figure()
    plt.scatter(X[keys[ii[i]]], Y)
    plt.show()

    return None

def training_datos_2018(val_curul, ii, modelo):
    df = pd.read_csv("senado2018.csv")
    keys = df.keys()

    partidos = np.array(df[keys[0]])
    curules = np.array(df[keys[2]])

    up_val = curules > val_curul
    down_val = curules <= val_curul
    
    Y = np.ones(len(partidos), dtype=int)
    Y[up_val] = 1
    Y[down_val] = 0

    print("\nY 2018: ", Y, sep="\n")

    X = df[keys[ii]]
    #print("\n", X)
    
    modelo.fit(X, Y)
    
    Y_predict = modelo.predict(X)
    print("\nY training 2018 predict: ", Y_predict, sep="\n")  

    print("\nInformacion coeficientes")
    print("\nClases: ", modelo.classes_)
    print("\nCoeficientes: ", modelo.coef_)
    print("\nIntercepto: ", modelo.intercept_)

    print("\nMetricas")    

    precision0 = sklearn.metrics.precision_score(Y, Y_predict, pos_label=0)
    print("precision 0: ", precision0)

    precision1 = sklearn.metrics.precision_score(Y, Y_predict, pos_label=1) 
    print("precision 1: ", precision1)

    recall0 = sklearn.metrics.recall_score(Y, Y_predict, pos_label=0) 
    print("recall 0: ", recall0)

    recall1 = sklearn.metrics.recall_score(Y, Y_predict, pos_label=1)
    print("recall 1: ", recall1)

    f1_score0 = sklearn.metrics.f1_score(Y, Y_predict, pos_label=0)
    print("f1 score 0: ", f1_score0)

    f1_score1 = sklearn.metrics.f1_score(Y, Y_predict, pos_label=1)
    print("f1 score 1: ", f1_score1)

    proba = modelo.predict_proba(X)
    #print(proba)

    return None

def test_datos_2022(val_curul, ii, modelo):
    df2 = pd.read_csv("senado2022.csv")
    keys2 = df2.keys()

    partidos2 = np.array(df2[keys2[0]])
    curules2 = np.array(df2[keys2[2]])

    up_val2 = curules2 > val_curul
    down_val2 = curules2 <= val_curul

    Y2 = np.ones(len(partidos2), dtype=int)
    Y2[up_val2] = 1
    Y2[down_val2] = 0

    print("\nY 2022: ", Y2, sep="\n")

    X2 = df2[keys2[ii]]

    Y2_predict = modelo.predict(X2)
    print("\nY test 2022 predict: ", Y2_predict, sep="\n")
    
    print("\nInformacion coeficientes")
    print("\nClases: ", modelo.classes_)
    print("\nCoeficientes: ", modelo.coef_)
    print("\nIntercepto: ", modelo.intercept_)
    
    print("\nMetricas")

    precision0 = sklearn.metrics.precision_score(Y2, Y2_predict, pos_label=0)
    print("precision 0: ", precision0)

    precision1 = sklearn.metrics.precision_score(Y2, Y2_predict, pos_label=1) 
    print("precision 1: ", precision1)

    recall0 = sklearn.metrics.recall_score(Y2, Y2_predict, pos_label=0) 
    print("recall 0: ", recall0)

    recall1 = sklearn.metrics.recall_score(Y2, Y2_predict, pos_label=1)
    print("recall 1: ", recall1)

    f1_score0 = sklearn.metrics.f1_score(Y2, Y2_predict, pos_label=0)
    print("f1 score 0: ", f1_score0)

    f1_score1 = sklearn.metrics.f1_score(Y2, Y2_predict, pos_label=1)
    print("f1 score 1: ", f1_score1)

    proba = modelo.predict_proba(X2)
    #print(proba)
    
    return None


val_curul = 13
ii = [3, 4, 5, 6, 7, 8]         #ii = [3, 4, 5, 6, 7, 8]
modelo = sklearn.linear_model.LogisticRegression(max_iter=10000)

print("Training datos 2018")
training_datos_2018(val_curul, ii, modelo)

print("\nTest datos 2022")
test_datos_2022(val_curul, ii, modelo)














