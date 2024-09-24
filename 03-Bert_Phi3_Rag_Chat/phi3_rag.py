from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

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

def generate_answer(text_generator, tokenizer, index, embedder, sentences, query):
    results = search(index, embedder, sentences, query)
    context = " ".join(results)

    messages = [ 
        {"role": "system", "content": f"Você é um especialista ferroviário. Contexto: {context}"}, 
        {"role": "user", "content": query}
    ] 

    pipe = pipeline( 
        "text-generation", 
        model=text_generator, 
        tokenizer=tokenizer
    ) 

    generation_args = { 
        "max_new_tokens": 500, # Número máximo de tokens a serem gerados. Em outras palavras, o tamanho da sequência de saída, não incluindo os tokens do prompt
        "return_full_text": False, 
        "temperature": 0.1, 
        "top_p": 0.3,
        "do_sample": True
    } 

    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

# Carrega e indexa o texto ao iniciar o servidor
txt_path = "rof.txt"
sentences = load_text_from_txt(txt_path)
index, sentences, embedder = index_text(sentences)

# Inicializa modelo de geração de texto
torch.random.manual_seed(0) 
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 
text_generator = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get("question")
    if not query:
        return jsonify({"error": "A pergunta não foi fornecida"}), 400
    
    answer = generate_answer(text_generator, tokenizer, index, embedder, sentences, query)
    
    return jsonify({"answer": answer, "score": 0.99})

if __name__ == '__main__':
    app.run(debug=True)
