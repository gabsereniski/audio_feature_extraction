# Classificação de canto de pássaros

### Time

|  RA   | Name |
| -------- | ------- |
| 2349256    |  Ana Carla Quallio Rosa  |
| 2349345 |     Gabriela Paola Sereniski |
| 2046407    |  Maria Fernanda Pinguelli Sodré  |
| 2252813  |     Reginaldo Gregório de Souza Neto |


### Extração de características de áudio

Funções para extração das características espectrais de sinais de áudio em python.

A base de dados consiste nas gravações das 10 espécies de pássaros com maior volume de amostras da competição Birdclef 2023.

Execução: `python extractor.py`

### Separação das características
Após a extração execute `python separacao_features.py` para escolher quantos % da base você quer destinar para treino/teste

### Execução da classificação
Em seguida basta executar `python exampleMLP.py` para testar com os algoritmos de classificação disponibilizados pelo professor para obter as acurácias.
