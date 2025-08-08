import json
import chromadb
from langchain.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM  # Updated import
from langchain.vectorstores import Chroma

# Initialize the embedding model and ChromaDB client once
embedding_model = OllamaEmbeddings(model="llama3.2:1b")
client = chromadb.PersistentClient(path="./Mech_Bot/chroma_db")

def load_wikipedia_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_chroma_db(data):
    collection = client.get_or_create_collection(name="mechanical_engineering")
    
    for topic, content in data.items():
        embedding = embedding_model.embed_documents([content['full_text']])[0]
        collection.add(
            documents=[content['full_text']],
            embeddings=[embedding],
            metadatas=[{"topic": topic}],
            ids=[topic]
        )
    
    return collection

def query_rag(question, collection):
    query_embedding = embedding_model.embed_query(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_docs = "\n".join([doc for doc in results['documents'][0]])
    
    llm = OllamaLLM(model="llama3.2:1b")  # Updated class
    prompt = f"""
    Use the following retrieved context to answer the question.
    Context:
    {retrieved_docs}
    
    Question: {question}
    Answer:
    """
    
    response = llm.generate(prompts=[prompt])
    answer = response.generations[0][0].text
    return answer

if __name__ == "__main__":
    data = load_wikipedia_data("Mech_Bot\mechanical_engineering_wikipedia.json")
    collection = setup_chroma_db(data)
    
    while True:
        user_query = input("Ask a question (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        answer = query_rag(user_query, collection)
        print("\nResponse:\n", answer)