#MÓDULO DE TREINAMENTO - Cluster KMeans na base Boston Housing
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from pickle import dump

#1. ABRIR OS DADOS
dados = pd.read_csv('HousingData.csv', sep=',')
print('Registros:', dados.shape[0], '| Colunas:', dados.shape[1])

#2. TRATAR VALORES AUSENTES
#A base tem 20 valores vazios (NaN) em CRIM, ZN, INDUS, CHAS, AGE e LSTAT.
#O KMeans não aceita NaN. Apagar as linhas jogaria fora 112 das 506 casas,
#então preenchemos: numéricos com a mediana, categórico com a moda.
print('Valores ausentes por coluna:')
print(dados.isna().sum()[dados.isna().sum() > 0])

#3. SEPARAR CATEGÓRICOS E NUMÉRICOS
#CHAS: 1 = casa às margens do rio Charles, 0 = não. É categórico (sim/não).
#RAD é um índice de acesso a rodovias (1 a 8 e 24). Tratamos como numérico
#porque é ordinal (maior = mais acesso) e virar 9 colunas dummy faria o RAD
#dominar o cálculo de distância.
colunas_cat = ['CHAS']
colunas_num = [c for c in dados.columns if c not in colunas_cat]

dados[colunas_num] = dados[colunas_num].fillna(dados[colunas_num].median())
dados['CHAS'] = dados['CHAS'].fillna(dados['CHAS'].mode()[0]).astype(int)

dados_num = dados[colunas_num]
dados_cat = dados[colunas_cat]

#Não há necessidade de balancear: no cluster não existe classe a prever.

#4. NORMALIZAR OS DADOS NUMÉRICOS (MinMax: tudo entre 0 e 1)
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_num)
#Salvar para normalizar a nova residência e desnormalizar o centróide
dump(modelo_normalizador, open('normalizador_housing.model', 'wb'))

dados_num_normalizados = modelo_normalizador.transform(dados_num)  #transform, não fit_transform de novo
dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=colunas_num)

#5. NORMALIZAR OS DADOS CATEGÓRICOS (get_dummies)
#CHAS vira duas colunas 0/1: CHAS_0 e CHAS_1
dados_cat_normalizados = pd.get_dummies(dados_cat['CHAS'], prefix='CHAS', dummy_na=False, dtype=int)

#Juntar numéricos e categóricos num dataframe único
dados_normalizados = dados_num_normalizados.join(dados_cat_normalizados)
print(dados_normalizados.head())

#6. NÚMERO ÓTIMO DE CLUSTERS - MÉTODO DO COTOVELO (distâncias)
#Testamos k de 1 a 20. O intervalo influencia o resultado: com uma faixa
#muito grande (ex.: 1 a 99) a reta do cotovelo fica inclinada demais e puxa o k para cima.
distorcoes = []
K = range(1, 21)
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42, n_init=10).fit(dados_normalizados)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_normalizados, cluster_model.cluster_centers_, 'euclidean'),
                axis=1) / dados_normalizados.shape[0]
        )
    )

#Ponto mais distante da reta que liga o primeiro e o último ponto da curva
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distancias.append(numerador / denominador)

n_clusters_otimo = K[distancias.index(max(distancias))]
print('Número ótimo de clusters:', n_clusters_otimo)

#Gráfico opcional do cotovelo (descomente para ver)
# from matplotlib import pyplot as plt
# plt.plot(K, distorcoes, marker='o')
# plt.axvline(n_clusters_otimo, color='red', linestyle='--')
# plt.xlabel('n clusters'); plt.ylabel('distorção'); plt.grid(); plt.show()

#7. TREINAMENTO DEFINITIVO E SALVAR O MODELO
housing_cluster = KMeans(n_clusters=n_clusters_otimo, random_state=42, n_init=10).fit(dados_normalizados)
dump(housing_cluster, open('housing_cluster.model', 'wb'))
print('Modelo salvo em housing_cluster.model')

#Quantas casas caíram em cada cluster
print(pd.Series(housing_cluster.labels_).value_counts().sort_index())
