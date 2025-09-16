import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk
from nltk import tokenize
import seaborn as sns
import matplotlib.pyplot as plt

def classificar_texto(df_name, avaliacao, sentimento):
  
    vetorizar = CountVectorizer(max_features=50)
    bag_of_words = vetorizar.fit_transform(df_name[avaliacao])

    X_treino, X_teste, y_treino, y_teste = train_test_split(bag_of_words, df_name[sentimento])

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(X_treino, y_treino)
    acuracia = regressao_logistica.score(X_teste, y_teste)
    return acuracia

def grafico_frequencia(df_name, coluna, qttd):
    todas_as_palavras = ' '.join(df_name[coluna])
    
    token_espaco = tokenize.WhitespaceTokenizer()
    token_frase = token_espaco.tokenize(todas_as_palavras)
    frequencia = nltk.FreqDist(token_frase)
    df_frequencia = pd.DataFrame({'Palavra': list(frequencia.keys()),
                              'Frequência': list(frequencia.values())})
    df_frequencia.nlargest(columns='Frequência', n=qttd)

    plt.figure(figsize=(20,6))
    ax = sns.barplot(data=df_frequencia.nlargest(columns='Frequência', n=20), x='Palavra', y='Frequência', color='gray')
    ax.set(ylabel='Contagem')
    plt.show()


df = pd.read_csv('https://raw.githubusercontent.com/alura-cursos/nlp_analise_sentimento/refs/heads/main/Dados/dataset_avaliacoes.csv')

acc = classificar_texto(df, 'avaliacao', 'sentimento')
print(f'{acc*100}% de acurácia')

grafico_frequencia(df, 'avaliacao', 5)
