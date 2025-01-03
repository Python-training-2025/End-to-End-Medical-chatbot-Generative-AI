from  src.helper import  load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


# Recharge manuellement le fichier .env
load_dotenv(override=True)
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file(data="data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


# Initialisation de la connexion à Pinecone avec la clé API stockée dans les variables d'environnement.
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment="us-east-1",  # Remplacez par votre région
    timeout=180  # Spécifiez ici le délai d'attente
)

# Nom de l'index à créer dans la base de données vectorielle Pinecone.
index_name = "medicalbot"

# Création d'un nouvel index dans Pinecone avec des paramètres spécifiques.
pc.create_index(
    name=index_name,  # Nom de l'index utilisé pour identifier et stocker les vecteurs.
    dimension=384,  # Dimension des vecteurs (à remplacer par la dimension réelle du modèle, par exemple 384).
    metric="cosine",  # Métrique utilisée pour mesurer la similarité entre vecteurs (cosine dans ce cas).
    spec=ServerlessSpec(  # Configuration du déploiement serverless (sans gestion explicite du serveur).
        cloud="aws",  # Fournisseur cloud où l'index est hébergé (ici Amazon Web Services).
        region="us-east-1"  # Région géographique pour optimiser la latence et la disponibilité.
    )
)

# Intégrer chaque segment et insérer les embeddings dans votre index Pinecone.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
