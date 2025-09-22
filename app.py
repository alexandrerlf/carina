import joblib

# carregar artefatos
modelo = joblib.load("modelo_logreg.pkl")
vetorizador = joblib.load("vetorizador_tfidf.pkl")
encoder = joblib.load("label_encoder.pkl")

# exemplo de novo texto
novo_texto = ["I have a problem with my credit card payment"]

# transformar texto em vetores
X_novo = vetorizador.transform(novo_texto)

# prever classe (número)
pred_num = modelo.predict(X_novo)

# converter número para nome real
pred_classe = encoder.inverse_transform(pred_num)

print("Classe prevista:", pred_classe[0])
