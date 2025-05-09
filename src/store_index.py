from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file(data = 'Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

#Embed each chunk and upsert the embeddings into your pinecone index

docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name=index_name,
    embedding=embeddings
)

#before launching this application the first step is to execute the store_index.py first as this is the starting pint for this after executing store_index.py we will launch app.py
#and store_index.py will be exceuted only once but if you are adding additional data then we need to execute it again