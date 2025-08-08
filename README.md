# 🚗 CarSenseAI - AI-Powered Mechanical Engineering Assistant

[![GitHub Repository](https://img.shields.io/badge/GitHub-CarSenseAI-blue?logo=github)](https://github.com/aadilshaikh123/CarSensAI)
[![Python](https://img.shields.io/badge/Python-3.8+-green?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)

A sophisticated Retrieval-Augmented Generation (RAG) system fine-tuned on automotive and mechanical engineering knowledge, providing accurate answers to technical questions about cars, engines, and mechanical systems.

## 🎯 Project Overview

This project combines fine-tuned LLaMA 3.2 1B model with a vector database to create an intelligent Q&A bot that can answer complex mechanical engineering and automotive questions. The system uses LoRA (Low-Rank Adaptation) for efficient fine-tuning and ChromaDB for semantic search capabilities.

## 🏗️ Architecture

```
## 🏗️ Architecture

![CarSenseAI Architecture](./G:\Mech_Bot\architecture.png)

*CarSenseAI system architecture showing the complete pipeline from data sources to user interface*
```

## 📁 Project Structure

### Core Application Files
- **`streamlit_app.py`** - Main web interface with chat functionality
- **`finetune.py`** - Model fine-tuning script using LoRA/QLoRA
- **`bot.py`** - Alternative CLI-based bot implementation

### Data Processing
- **`scrape.py`** - Wikipedia scraper for mechanical engineering topics
- **`scrape_articles.py`** - 2CarPros Q&A scraper for automotive questions
- **`preprocess.ipynb`** - Data cleaning and preprocessing notebook
- **`l.py`** - Base model downloader script

### Data Files (Essential)
- **`small_car_dataset.jsonl`** - Curated dataset for Streamlit app (17 Q&A pairs)
- **`dataset.jsonl`** - Main training dataset for fine-tuning (276 entries)

### Model & Database Directories
- **`llama-3.2-1b-base/`** - Base LLaMA model files
- **`llama-3.2-1b-instruct-cars-finetuned-adapter/`** - Fine-tuned LoRA adapter
- **`chroma_db/`** - Original vector database
- **`chroma_db_streamlit_finetuned/`** - Vector database for Streamlit app

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
# CUDA-enabled GPU (recommended for fine-tuning)
# 16GB+ RAM recommended
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/aadilshaikh123/CarSenseAI.git
cd CarSenseAI
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft trl bitsandbytes
pip install streamlit chromadb pandas scikit-learn
pip install requests beautifulsoup4 datasets
```

4. **Set up Hugging Face token** (for model access)
```bash
export HF_TOKEN="your_huggingface_token_here"  # Linux/Mac
set HF_TOKEN=your_huggingface_token_here       # Windows
```

## 📖 Usage

### 1. Running the Streamlit App (Recommended)

```bash
streamlit run streamlit_app.py
```

The app will:
- Load the fine-tuned model and tokenizer
- Set up the vector database with embeddings
- Provide a chat interface for Q&A

**Features:**
- Interactive chat interface
- Context retrieval display
- Real-time response generation
- Fine-tuned model integration

### 2. Fine-tuning the Model

```bash
python finetune.py
```

This will:
- Load the base LLaMA 3.2-1B model
- Apply LoRA fine-tuning on the dataset
- Save the adapter to `llama-3.2-1b-instruct-cars-finetuned-adapter/`

**Fine-tuning Configuration:**
- **LoRA Rank**: 16
- **Alpha**: 32
- **Dropout**: 0.05
- **Batch Size**: 2 (with gradient accumulation)
- **Learning Rate**: 2e-5
- **4-bit Quantization**: Enabled (QLoRA)

### 3. Data Collection & Processing

**Scrape Wikipedia Data:**
```bash
python scrape.py
```

**Scrape Automotive Q&A:**
```bash
python scrape_articles.py
```

**Process Raw Data:**
```bash
jupyter notebook preprocess.ipynb
```

### 4. CLI Bot (Alternative)

```bash
python bot.py
```

## 🎯 Key Features

### 🤖 Fine-tuned Language Model
- **Base Model**: LLaMA 3.2-1B Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit QLoRA for memory efficiency
- **Domain**: Automotive and mechanical engineering

### 🔍 Retrieval-Augmented Generation (RAG)
- **Vector Database**: ChromaDB
- **Embedding Model**: Model's own hidden states
- **Retrieval**: Top-3 most relevant documents
- **Context Integration**: Seamless prompt augmentation

### 💬 User Interface
- **Framework**: Streamlit
- **Features**: Chat history, context visualization, real-time responses
- **Responsive**: Works on desktop and mobile

### 📊 Data Sources
- **Wikipedia**: Mechanical engineering articles
- **2CarPros**: Automotive Q&A community
- **Custom Datasets**: Curated mechanical engineering questions

## 🔧 Configuration

### Model Parameters
```python
# In finetune.py
base_model_id = "meta-llama/Llama-3.2-1b-Instruct"
lora_r = 16
lora_alpha = 32
learning_rate = 2e-5
max_seq_length = 1024
```

### Streamlit App Settings
```python
# In streamlit_app.py
DATASET_PATH = "small_car_dataset.jsonl"
CHROMA_DB_PATH = "./chroma_db_streamlit_finetuned"
FINETUNED_MODEL_PATH = "llama-3.2-1b-instruct-cars-finetuned-adapter"
```

## 📈 Performance

### Model Capabilities
- **Domain Knowledge**: Automotive systems, engines, maintenance
- **Response Quality**: Technical accuracy with clear explanations
- **Speed**: Fast inference with 4-bit quantization
- **Memory Usage**: ~4-6GB GPU memory for inference

### RAG System
- **Retrieval Accuracy**: High relevance with semantic search
- **Context Length**: Up to 512 tokens per retrieved document
- **Response Time**: <5 seconds for most queries

## 🛠️ Development

### Adding New Data
1. Add Q&A pairs to `small_car_dataset.jsonl` for Streamlit
2. Add training data to `dataset.jsonl` for fine-tuning
3. Re-run fine-tuning: `python finetune.py`
4. Restart Streamlit app

### Modifying the Model
1. Update configuration in `finetune.py`
2. Adjust LoRA parameters for different performance/speed trade-offs
3. Modify chat template formatting in `format_chat_template()`

### Extending Functionality
- Add new data sources in scraping scripts
- Implement additional preprocessing in `preprocess.ipynb`
- Customize UI components in `streamlit_app.py`

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```python
# Reduce batch size in finetune.py
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
```

**Model Loading Errors:**
```bash
# Ensure HF_TOKEN is set
export HF_TOKEN="your_token"
# Check model path exists
ls llama-3.2-1b-instruct-cars-finetuned-adapter/
```

**Streamlit Connection Issues:**
```bash
# Clear cache and restart
streamlit cache clear
streamlit run streamlit_app.py
```

## 📄 License

This project is for educational and research purposes. Please ensure compliance with:
- LLaMA 2 License Agreement
- Hugging Face Model License
- Data source terms of service

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration settings
3. Ensure all dependencies are installed
4. Verify GPU/CUDA setup for fine-tuning

## 🎯 Future Enhancements

- [ ] Multi-modal support (images, diagrams)
- [ ] Voice interface integration
- [ ] Advanced retrieval methods (re-ranking)
- [ ] Model quantization optimization
- [ ] Deployment to cloud platforms
- [ ] API endpoint creation
- [ ] Performance monitoring dashboard

---

**Built with ❤️ for the mechanical engineering and automotive community**
