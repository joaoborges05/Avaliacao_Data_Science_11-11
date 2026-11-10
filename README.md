# Avaliacao_Data_Science_11-11
Avaliação bimestral da matéria de Data Science de 2026

## Cluster KMeans na base Boston Housing

Agrupa as 506 residências da base `HousingData.csv` com KMeans e descobre a qual grupo pertence uma nova residência.

| Arquivo | O que faz |
|---|---|
| `treino_cluster_housing.py` | Trata os valores ausentes, normaliza os dados, escolhe o número de clusters pelo método do cotovelo, treina o modelo e salva `housing_cluster.model` e `normalizador_housing.model` |
| `inferencia_housing.py` | Carrega os dois modelos, informa o cluster de uma nova residência e mostra o centróide desnormalizado |
| `HousingData.csv` | Base de dados (506 registros, 14 colunas) |

### Como rodar

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python treino_cluster_housing.py
python inferencia_housing.py
```

Rode o treino antes da inferência, porque é ele que gera os arquivos `.model`.

Resultado esperado: 6 clusters (com 34, 183, 35, 110, 89 e 55 casas). A residência de exemplo cai no cluster 1.
