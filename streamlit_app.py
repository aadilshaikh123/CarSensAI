import streamlit as st
import json
import chromadb
import os
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Configuration ---
DATASET_PATH = "small_car_dataset.jsonl"
CHROMA_DB_PATH = "./chroma_db_streamlit_finetuned"
COLLECTION_NAME = "car_qna_finetuned"
FINETUNED_MODEL_PATH = "llama-3.2-1b-instruct-cars-finetuned-adapter"

# --- Helper Functions ---

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

@st.cache_resource
def load_finetuned_model_and_tokenizer(model_path):
    """Loads the fine-tuned model and tokenizer."""
    print(f"Loading fine-tuned model and tokenizer from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set tokenizer pad_token to eos_token")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if DEVICE.type == 'cuda' else torch.float32
        ).to(DEVICE)
        model.eval()
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading fine-tuned model/tokenizer: {e}")
        print(f"Error loading fine-tuned model/tokenizer: {e}")
        return None, None

@st.cache_resource
def get_text_generation_pipeline(_model, _tokenizer):
    """Creates a text generation pipeline."""
    print("Creating text generation pipeline...")
    try:
        generator = pipeline(
            "text-generation", 
            model=_model, 
            tokenizer=_tokenizer, 
            device=DEVICE,
            max_new_tokens=150,
            pad_token_id=_tokenizer.pad_token_id
        )
        print("Text generation pipeline created.")
        return generator
    except Exception as e:
        st.error(f"Error creating text generation pipeline: {e}")
        print(f"Error creating text generation pipeline: {e}")
        return None

def get_embedding(text, model, tokenizer):
    """Generates embedding using mean pooling of last hidden states."""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error generating embedding for text '{text[:50]}...': {e}")
        return None

@st.cache_resource
def get_chroma_client():
    """Initializes and returns the ChromaDB client."""
    print("Initializing ChromaDB client...")
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

def load_data(filename):
    """Loads Q&A data from a JSONL file."""
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if line.startswith('[') or line.endswith(']'):
                        line = line.strip('[],')
                    if line:
                        data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")
    except FileNotFoundError:
        st.error(f"Error: Dataset file not found at {filename}")
        return None
    return data

@st.cache_resource(show_spinner="Setting up Vector Database...")
def setup_chroma_db(_client, _model, _tokenizer, data):
    """Sets up the ChromaDB collection with embeddings."""
    print("Setting up ChromaDB collection...")
    collection = _client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 5:
        print(f"Collection '{COLLECTION_NAME}' already exists and seems populated. Skipping setup.")
        return collection

    print(f"Populating collection '{COLLECTION_NAME}'...")
    texts_to_embed = []
    embeddings_list = []
    metadatas = []
    ids = []

    for item in data:
        if 'text' in item and isinstance(item['text'], str):
            embedding = get_embedding(item['text'], _model, _tokenizer)
            if embedding is not None:
                texts_to_embed.append(item['text'])
                embeddings_list.append(embedding.tolist())
                metadatas.append({"source": os.path.basename(DATASET_PATH)})
                ids.append(str(uuid.uuid4()))
            else:
                print(f"Skipping item due to embedding error: {item}")
        else:
            print(f"Skipping invalid data item: {item}")

    if not texts_to_embed:
        st.warning("No valid text entries found or embedded in the dataset.")
        return collection

    try:
        if texts_to_embed and embeddings_list and metadatas and ids:
            collection.add(
                documents=texts_to_embed,
                embeddings=embeddings_list,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(ids)} documents to collection.")
        else:
            print("No documents were added to the collection.")
    except Exception as e:
        st.error(f"Error adding documents to ChromaDB: {e}")
        print(f"Error during ChromaDB setup (add step): {e}")

    return collection

def query_rag(question, _collection, _model, _tokenizer, _generator):
    """Performs RAG to answer a question."""
    if not _collection:
        return "Error: Vector database collection is not available.", ""
    if not _generator:
        return "Error: Text generation pipeline is not available.", ""

    try:
        query_embedding = get_embedding(question, _model, _tokenizer)
        if query_embedding is None:
            return "Error generating query embedding.", ""
            
        results = _collection.query(query_embeddings=[query_embedding.tolist()], n_results=3)

        retrieved_docs = "\n\n".join([doc for doc in results['documents'][0]])

        prompt = f"""
        You are a helpful assistant knowledgeable about cars and mechanical topics.
        Use the following retrieved context to answer the question accurately and concisely.
        If the context doesn't contain the answer, say you don't have enough information from the provided context.

        Context:
        {retrieved_docs}

        Question: {question}

        Answer:
        """

        pipeline_output = _generator(prompt)
        answer = pipeline_output[0]['generated_text']
        answer = answer.replace(prompt, "").strip()
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()

        return answer, retrieved_docs
    except Exception as e:
        st.error(f"Error during RAG query: {e}")
        print(f"Error during RAG query: {e}")
        return f"An error occurred: {e}", ""


# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("🚗 Mech Q&A Bot (RAG - Fine-tuned Model)")
st.caption(f"Using fine-tuned model: {os.path.basename(FINETUNED_MODEL_PATH)}")

model, tokenizer = load_finetuned_model_and_tokenizer(FINETUNED_MODEL_PATH)
generator = None
if model and tokenizer:
    generator = get_text_generation_pipeline(model, tokenizer)

client = get_chroma_client()
dataset = load_data(DATASET_PATH)

collection = None
if dataset and client and model and tokenizer:
    collection = setup_chroma_db(client, model, tokenizer, dataset)
elif not dataset:
    st.error(f"Failed to load the dataset. Please ensure '{os.path.basename(DATASET_PATH)}' exists.")
    st.stop()
elif not model or not tokenizer:
    st.error("Failed to load the fine-tuned model or tokenizer. Please check the path and model files.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["role"] == "assistant":
            with st.expander("Retrieved Context"):
                st.text(message["context"])

if prompt := st.chat_input("Ask a question about cars or mechanics:"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if collection is not None and generator is not None and model is not None and tokenizer is not None:
        answer, context = query_rag(prompt, collection, model, tokenizer, generator)
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Retrieved Context"):
                st.text(context if context else "No context retrieved.")
        st.session_state.messages.append({"role": "assistant", "content": answer, "context": context})
    else:
        error_message = "Could not generate response. Dependencies missing: "
        missing = []
        if collection is None: missing.append("Vector DB")
        if generator is None: missing.append("Text Generator")
        if model is None or tokenizer is None: missing.append("Model/Tokenizer")
        error_message += ", ".join(missing) + "."
        
        with st.chat_message("assistant"):
            st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})

