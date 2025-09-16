import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def classificar_texto(df_name, avaliacao, sentimento):
  
    vetorizar = CountVectorizer(max_features=50)
    bag_of_words = vetorizar.fit_transform(avaliacao)

    X_treino, X_teste, y_treino, y_teste = train_test_split(bag_of_words, sentimento)

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(X_treino, y_treino)
    acuracia = regressao_logistica.score(X_teste, y_teste)
    return acuracia


df = pd.read_csv('https://raw.githubusercontent.com/alura-cursos/nlp_analise_sentimento/refs/heads/main/Dados/dataset_avaliacoes.csv')

acc = classificar_texto(df, df['avaliacao'], df['sentimento'])
print(acc)