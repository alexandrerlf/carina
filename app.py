import joblib

modelo = joblib.load("modelo_logreg.pkl")
vetorizador = joblib.load("vetorizador_tfidf.pkl")

novo_texto = ["I have a problem with my credit card payment"]
X_novo = vetorizador.transform(novo_texto)
print(modelo.predict(X_novo))
