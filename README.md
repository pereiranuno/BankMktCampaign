# Projeto de Fundamentos Machine Learning
# The Rumos Bank Marketing Campaign

## Descrição

Este repositório contém um notebook Jupyter desenvolvido no âmbito do projeto do Módulo de Fundamentos de Machine Learning. O objetivo é entregar um modelo preditivo de classificação, explorando diferentes algoritmos e ajustando os seus hiperparâmetros.

## Requisitos

Para executar este projeto, certifica-te de que tens as seguintes bibliotecas instaladas:

```python
from google.colab import drive
from google.colab import drive
from collections import defaultdict
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  precision_recall_curve, roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score,auc, roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder,LabelEncoder
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
```


## Como utilizar

1. Clona este repositório:

```sh
git clone https://github.com/pereiranuno/BankMktCampaign/
cd BankMktCampaign
```


2. Abre o Jupyter Notebook:

```sh
jupyter notebook np_final_project_(5).ipynb
```

4. Explora o código e os resultados no notebook.

## Autor

Este projeto foi desenvolvido por Nuno Pereira.

