from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


# Importation et initialisation de l'application Flask pour créer un serveur web.
app = Flask(__name__)

# Chargement des variables d'environnement depuis un fichier .env.
load_dotenv()

# Récupération des clés API stockées dans les variables d'environnement.
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# Configuration des clés API dans les variables d'environnement pour les utiliser dans le code.
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Téléchargement et chargement des embeddings depuis Hugging Face pour la correspondance sémantique.
embeddings = download_hugging_face_embeddings()

# Nom de l'index dans Pinecone pour stocker ou récupérer les embeddings.
index_name = "medicalbot"

# Chargement d'un index existant dans Pinecone avec les embeddings fournis pour effectuer des recherches.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Création d'un récupérateur pour rechercher les 3 documents les plus similaires en utilisant la similarité sémantique.
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialisation du modèle OpenAI avec des paramètres personnalisés (température et taille des tokens).
llm = OpenAI(temperature=0.4, max_tokens=500)

# Création d'un modèle de prompt structuré pour guider la génération des réponses par l'assistant.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # Prompt système pour définir les règles d'interaction.
        ("human", "{input}"),       # Placeholder pour insérer l'entrée utilisateur.
    ]
)

# Création d'une chaîne de traitement des documents combinant le modèle et le prompt.
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Création d'une chaîne RAG (Retrieval-Augmented Generation) pour combiner récupération et génération.
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Définition de la route principale pour afficher la page HTML du chatbot.
@app.route("/")
def index():
    return render_template("chat.html")

# Définition de la route pour gérer les requêtes envoyées par l'utilisateur et retourner des réponses.
@app.route("/get", methods=["GET", "POST"])
def chat():
    # Récupération du message envoyé par l'utilisateur depuis le formulaire HTML.
    msg = request.form["msg"]
    input = msg
    print(input)

    # Invocation de la chaîne RAG pour traiter la question et générer une réponse basée sur les données récupérées.
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])  # Affiche la réponse dans la console.

    # Retourne la réponse générée au client (navigateur).
    return str(response["answer"])

# Démarrage de l'application Flask sur l'hôte local avec débogage activé.
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8083, debug= True)
