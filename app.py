from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import openai
import faiss
from openai.embeddings_utils import get_embedding
import os


# Establece la clave de la API para el servicio OpenAI
openai.api_key = os.getenv("OPENAI_KEY")

app = Flask(__name__)
CORS(app)

MODEL = "gpt-3.5-turbo"

# Carga los datos y crea un índice Faiss si no existe
def load_data():
    try:
        embeddings = np.load('embeddings.npy')
        index = faiss.read_index('faiss.index')
        df = pd.read_csv("embedding4.csv", index_col=0)
    except (FileNotFoundError, IOError):
        df = pd.read_csv("embedding4.csv", index_col=0)
        df["embedding"] = df.embedding.apply(eval).apply(np.array)
        embeddings = np.vstack(df["embedding"].values)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        np.save('embeddings.npy', embeddings)
        faiss.write_index(index, 'faiss.index')

    return embeddings, index, df

embeddings, index, df = load_data()

# Función para buscar revisiones
def search_reviews(prompt, k=1000):
    prompt_embedding = get_embedding(prompt, engine="text-embedding-ada-002")
    D, I = index.search(np.array([prompt_embedding]), k)
    results = df.iloc[I[0]]
    results['distances'] = D[0]
    # Filtrar los resultados donde la distancia es hasta 0.41 y 'Page Number' no es igual a 1, 2 o 3
    results = results[(results['distances'] <= 0.41) & ~results['Page Number'].isin([1, 2, 3])]
    return results

def create_chat(query, page_text):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Eres un Abogado en Reglamentación Aeronáutica, Resume de manera breve como se relaciona la consulta del usuario con el Texto de la pagina. Tu respuesta debe debe tener maximo 30 palabras"},
            {"role": "user", "content": f"Pregunta del usuario: {query}"},
            {"role": "user", "content": f"Texto de la pagina: {page_text}"}
        ],
        temperature=0.5,
    )
    return response.choices[0].message['content']


@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')

    # Realiza la búsqueda si se ingresó un query
    results = search_reviews(query, k=1000)
    grouped_results = results.groupby(['Title', 'Link']).agg({'Page Number': list, 'Page Text': list, 'distances': 'mean'}).reset_index()
    grouped_results = grouped_results.sort_values('distances')

    # Prepara los resultados para enviarlos al cliente
    results_to_send = []
    for i, (index, row) in enumerate(grouped_results.iterrows()):
        page_numbers = row['Page Number']
        page_texts = row['Page Text']
        # Ordena las páginas y los textos de acuerdo a los números de página
        pages_and_texts = sorted(zip(page_numbers, page_texts), key=lambda x: x[0])
        first_page_number, first_page_text = pages_and_texts[0]

        # Calcula el chat response solo para la primera página de cada documento
#        chat_response = create_chat(query, first_page_text) if i < 2 else ''

        page_numbers_with_distance = [f"{page}" for page, text in pages_and_texts]
        results_to_send.append({
            'Title': row['Title'].replace('.pdf', ''),
            'Page Number': page_numbers_with_distance,
#            'Chat Response': chat_response,
            'Link': row['Link']
        })

    return jsonify(results_to_send)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

