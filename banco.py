from pymongo import MongoClient
import pandas as pd
import os

# Conectar ao MongoDB local
client = MongoClient(os.environ.get("MONGO_URI"))

# Criar banco de dados e coleção
db = client["carina"]
collection = db["dados"]

class Banco:
    def inserir_dados(self, dados):
        if isinstance(dados, pd.DataFrame):
            records = dados.to_dict(orient='records')
            collection.insert_many(records)
        else:
            raise ValueError("Os dados devem ser um DataFrame do pandas.")

    def buscar_dados(self, filtro=None):
        if filtro is None:
            filtro = {}
        dados = list(collection.find(filtro))
        return pd.DataFrame(dados)