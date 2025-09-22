import pandas as pd
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

# baixar recursos necessários
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

def limpar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'x{2,}', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def preprocessar_texto(texto):
    if pd.isnull(texto):
        return ""
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)
    tokens = word_tokenize(texto, language="english")
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def classificar_texto(df, coluna_texto, coluna_alvo, salvar_modelo=False):
    df[coluna_texto] = df[coluna_texto].astype(str).apply(preprocessar_texto)

    vetorizar = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vetorizar.fit_transform(df[coluna_texto])

    # LabelEncoder para o alvo
    le = LabelEncoder()
    y = le.fit_transform(df[coluna_alvo])

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial")
    modelo.fit(X_treino, y_treino)
    y_pred = modelo.predict(X_teste)

    acc = modelo.score(X_teste, y_teste)
    print(f"\nAcurácia: {acc*100:.2f}%\n")
    print("Relatório de Classificação:\n")
    print(classification_report(y_teste, y_pred, target_names=le.classes_))

    if salvar_modelo:
        joblib.dump(modelo, "data/modelo_logreg.pkl")
        joblib.dump(vetorizar, "data/vetorizador_tfidf.pkl")
        joblib.dump(le, "data/label_encoder.pkl")
        print("✅ Modelo, vetorizador e label encoder salvos!")

    return acc, vetorizar, modelo, le, y_teste, y_pred

def grafico_frequencia(df, coluna_texto, qttd=20):
    todas_palavras = " ".join(df[coluna_texto])
    tokens = todas_palavras.split()
    frequencia = nltk.FreqDist(tokens)
    df_frequencia = pd.DataFrame(
        {"Palavra": list(frequencia.keys()), "Frequência": list(frequencia.values())}
    )
    plt.figure(figsize=(20, 6))
    ax = sns.barplot(
        data=df_frequencia.nlargest(columns="Frequência", n=qttd),
        x="Palavra", y="Frequência", color="gray",
    )
    ax.set(ylabel="Contagem")
    plt.show()

def treino_modelo(dataset, coluna_alvo, salvar_modelo=False):
    if dataset.endswith(".json"):
        df = pd.read_json(dataset)
    else:
        df = pd.read_csv(dataset)

    df[coluna_alvo] = df[coluna_alvo].replace('Virtual currency', 
                                              'Money transfer, virtual currency, or money service')
    df[coluna_alvo] = df[coluna_alvo].replace('Money transfers', 
                                              'Money transfer, virtual currency, or money service')
    df[coluna_alvo] = df[coluna_alvo].replace('Credit card', 
                                              'Credit card or prepaid card')
    df[coluna_alvo] = df[coluna_alvo].replace('Prepaid card', 
                                              'Credit card or prepaid card')
    df[coluna_alvo] = df[coluna_alvo].replace('Credit reporting', 
                                              'Credit reporting, credit repair services, or other personal consumer reports')
    df[coluna_alvo] = df[coluna_alvo].replace('Payday loan', 
                                              'Payday loan, title loan, or personal loan')
    df[coluna_alvo] = df[coluna_alvo].replace('Payday loan, title loan', 
                                              'Payday loan, title loan, or personal loan')
    df[coluna_alvo] = df[coluna_alvo].replace('Other financial service', 
                                              'Bank account or service')
    

    df['Issue'] = df['Issue'].fillna('')
    df['Sub-issue'] = df['Sub-issue'].fillna('')
    df['Consumer complaint narrative'] = df['Consumer complaint narrative'].fillna('')

    df['texto_combinado'] = df.apply(
        lambda row: f"{row['Issue']} {row['Sub-issue']} {row['Consumer complaint narrative']}".strip(), axis=1
    )
    df['texto_combinado'] = df['texto_combinado'].apply(limpar_texto)

    _, _, _, encoder, y_teste, y_pred = classificar_texto(
        df, 'texto_combinado', coluna_alvo, salvar_modelo=salvar_modelo
    )

    grafico_frequencia(df, 'texto_combinado')

    cm = confusion_matrix(y_teste, y_pred)
    plt.figure(figsize=(10,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.show()

# Exemplo de treino e salvamento
treino_modelo("data/rows.csv", "Product", salvar_modelo=True)
