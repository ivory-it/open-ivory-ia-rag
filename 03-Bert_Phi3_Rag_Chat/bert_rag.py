from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

def load_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read().strip().split('\n')
    return text

def index_text(sentences):
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = embedder.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, sentences, embedder

def search(index, embedder, sentences, query, k=5):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)
    results = [sentences[i] for i in I[0]]
    return results

def generate_answer(question_answering, index, embedder, sentences, query):
    results = search(index, embedder, sentences, query)
    context = " ".join(results)
    print(f"\n\nContexto:\n{context}")
    answer = question_answering(question=query, context=context)
    return answer

# Carrega e indexa o texto ao iniciar o servidor
txt_path = "rof.txt"
sentences = load_text_from_txt(txt_path)
index, sentences, embedder = index_text(sentences)

# Inicializa pipeline com modelo de pergunta e reposta
question_answering = pipeline("question-answering", model="pierreguillou/bert-base-cased-squad-v1.1-portuguese")

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get("question")
    if not query:
        return jsonify({"error": "A pergunta não foi fornecida"}), 400
    
    answer = generate_answer(question_answering, index, embedder, sentences, query)
    
    return jsonify({"answer": answer['answer'], "score": answer['score']})

if __name__ == '__main__':
    app.run(debug=True)
