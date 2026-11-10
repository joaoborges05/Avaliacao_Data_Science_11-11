#MÓDULO DE INFERÊNCIA - a qual cluster pertence uma nova residência
import pandas as pd
from pickle import load

#Carregar os modelos salvos pelo treino_cluster_housing.py
housing_cluster = load(open('housing_cluster.model', 'rb'))
modelo_normalizador = load(open('normalizador_housing.model', 'rb'))

#1. RECEBER OS DADOS DE UMA NOVA RESIDÊNCIA (mesmas 14 colunas da base)
nova_residencia = pd.DataFrame([{
    'CRIM': 0.25,      #taxa de criminalidade per capita
    'ZN': 0.0,         #% de terrenos residenciais grandes
    'INDUS': 8.0,      #% de área industrial
    'CHAS': 0,         #1 = às margens do rio Charles
    'NOX': 0.50,       #concentração de óxido nítrico
    'RM': 6.2,         #nº médio de cômodos
    'AGE': 65.0,       #% de imóveis construídos antes de 1940
    'DIS': 4.0,        #distância até centros de emprego
    'RAD': 5,          #índice de acesso a rodovias
    'TAX': 300,        #imposto sobre a propriedade
    'PTRATIO': 18.0,   #alunos por professor
    'B': 390.0,        #índice demográfico da base original
    'LSTAT': 11.0,     #% de população de baixa renda
    'MEDV': 22.0,      #valor mediano do imóvel (US$ 1000)
}])

#2. NORMALIZAR A NOVA RESIDÊNCIA
#numéricos: o MESMO normalizador do treino, só transform (nunca fit de novo)
colunas_num = modelo_normalizador.feature_names_in_
nova_num_normalizada = pd.DataFrame(modelo_normalizador.transform(nova_residencia[colunas_num]), columns=colunas_num)

#categórico: get_dummies + reindex
#uma casa só gera 1 coluna dummy (CHAS_0 ou CHAS_1); o reindex cria a outra com 0
nova_cat_normalizada = pd.get_dummies(nova_residencia['CHAS'], prefix='CHAS', dummy_na=False, dtype=int)
nova_normalizada = nova_num_normalizada.join(nova_cat_normalizada).reindex(
    columns=housing_cluster.feature_names_in_, fill_value=0)

#3. INFERIR O CLUSTER (predict devolve o NÚMERO do cluster)
cluster_inferido = housing_cluster.predict(nova_normalizada)[0]
print('A nova residência pertence ao cluster:', cluster_inferido)

#4. DESNORMALIZAR O CENTRÓIDE DO CLUSTER INFERIDO
centroide = pd.DataFrame([housing_cluster.cluster_centers_[cluster_inferido]],
                         columns=housing_cluster.feature_names_in_)

#numéricos: inverse_transform volta para a escala original
centroide_num = pd.DataFrame(modelo_normalizador.inverse_transform(centroide[colunas_num]), columns=colunas_num)

#categórico: no centróide CHAS_0/CHAS_1 são PROPORÇÕES (ex.: 0.93 / 0.07)
#o from_dummies só aceita 0/1, então marca 1 na opção de maior proporção
centroide_cat = centroide.drop(columns=colunas_num)
centroide_cat_dummies = centroide_cat.eq(centroide_cat.max(axis=1), axis=0).astype(int)
centroide_cat_natural = pd.from_dummies(centroide_cat_dummies, sep='_')

centroide_natural = centroide_num.join(centroide_cat_natural)[nova_residencia.columns]

pd.set_option('display.width', 200)
print('\nNova residência:')
print(nova_residencia.round(2).to_string(index=False))
print('\nCentróide desnormalizado do cluster', cluster_inferido, ':')
print(centroide_natural.round(2).to_string(index=False))
print('\nProporção de casas à beira do rio (CHAS) no cluster:')
print((centroide_cat.round(2) + 0.0).to_string(index=False))  #+0.0 evita exibir -0.0
