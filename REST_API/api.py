# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
from gensim.models import KeyedVectors , Word2Vec
import os

model_path = "model/Reina_Valeria.bin"
app = Flask(__name__)

# Cargar el modelo una vez al inicio
model = None
try:
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Modelo cargado exitosamente")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {model_path}.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

@app.route('/')
def word2vec1():
    if model is None:
        return jsonify({"error": "El modelo no se cargó correctamente"}), 500
    
    word = request.args.get('word')
    if not word:
        return jsonify({"error": "No se proporcionó una palabra"}), 400

    try:
        similar_words = model.most_similar(positive=[word], topn=10)
        return jsonify(similar_words)
    except KeyError:
        return jsonify({"error": f"La palabra '{word}' no se encuentra en el vocabulario del modelo"}), 400

if __name__ == '__main__':
    app.run(debug=True)
