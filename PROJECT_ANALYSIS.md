# Project File Analysis - Mech Q&A Bot

## 📁 Essential Project Files

### Core Application Files (Keep in Git)
- **`streamlit_app.py`** - Main web interface with RAG system
- **`finetune.py`** - Model fine-tuning script using LoRA/QLoRA  
- **`bot.py`** - Alternative CLI bot implementation
- **`setup.py`** - Project setup and environment checker
- **`requirements.txt`** - Python dependencies
- **`README.md`** - Project documentation
- **`.gitignore`** - Git ignore rules

### Data Processing Scripts (Keep in Git)
- **`scrape.py`** - Wikipedia scraper for mechanical engineering topics
- **`scrape_articles.py`** - 2CarPros automotive Q&A scraper  
- **`l.py`** - Base model downloader utility
- **`preprocess.ipynb`** - Data cleaning and preprocessing notebook

### Essential Data Files (Keep in Git)
- **`small_car_dataset.jsonl`** - Curated Q&A dataset for Streamlit (17 pairs)
- **`dataset.jsonl`** - Main training dataset (276 entries)

## 🚫 Files to Exclude (.gitignore)

### Large Model Files (Generated/Downloaded)
- **`llama-3.2-1b-base/`** - Base LLaMA model (1GB+)
  - `config.json`, `model.safetensors`, `tokenizer.json`, etc.
- **`llama-3.2-1b-instruct-cars-finetuned-adapter/`** - Fine-tuned adapter
  - `adapter_model.safetensors`, training checkpoints, etc.

### Vector Databases (Generated)
- **`chroma_db/`** - Original ChromaDB database
- **`chroma_db_streamlit_finetuned/`** - Streamlit-specific database
- **`*.sqlite3`** - Database files

### Large Data Files (Raw/Processed)
- **`car_questions_live.json`** - Large scraped dataset (500MB+)
- **`car_questions_live - Copy.json`** - Backup copy
- **`mechanical_engineering_wikipedia.json`** - Wikipedia scrape data
- **`cleaned_output1.jsonl`** - Processed data file
- **`car_repair_train.jsonl`** - Training split (if generated)
- **`car_repair_val.jsonl`** - Validation split (if generated)

### Training Outputs
- **`runs/`** - TensorBoard logs
- **Training checkpoints** - Large temporary files

## 🔄 File Dependencies

### Data Flow
```
Raw Data Sources
    ↓
scrape.py & scrape_articles.py
    ↓
car_questions_live.json & mechanical_engineering_wikipedia.json
    ↓
preprocess.ipynb
    ↓
dataset.jsonl & small_car_dataset.jsonl
    ↓
finetune.py (uses dataset.jsonl)
    ↓
llama-3.2-1b-instruct-cars-finetuned-adapter/
    ↓
streamlit_app.py (uses small_car_dataset.jsonl + adapter)
```

### Model Pipeline
```
l.py → llama-3.2-1b-base/
dataset.jsonl + finetune.py → llama-3.2-1b-instruct-cars-finetuned-adapter/
streamlit_app.py → chroma_db_streamlit_finetuned/
```

## ⚙️ Setup Requirements

### Environment
1. **Python 3.8+**
2. **CUDA GPU** (recommended for fine-tuning)
3. **16GB+ RAM** (for model loading)
4. **Hugging Face Token** (for model access)

### Key Dependencies
- `torch` - PyTorch for ML
- `transformers` - Hugging Face models
- `peft` - LoRA fine-tuning
- `streamlit` - Web interface
- `chromadb` - Vector database
- `bitsandbytes` - Quantization

## 🎯 Usage Scenarios

### 1. Quick Demo (Recommended)
```bash
pip install -r requirements.txt
export HF_TOKEN="your_token"
streamlit run streamlit_app.py
```
Uses: `streamlit_app.py`, `small_car_dataset.jsonl`, pre-trained adapter

### 2. Full Training Pipeline
```bash
python scrape.py                    # Collect data
python scrape_articles.py          # More data  
jupyter notebook preprocess.ipynb  # Clean data
python finetune.py                 # Train model
streamlit run streamlit_app.py     # Run app
```

### 3. Development Mode
```bash
python setup.py                    # Check environment
python l.py                        # Download base model
python finetune.py                 # Custom training
streamlit run streamlit_app.py     # Test app
```

## 🔍 File Size Analysis

### Small Files (<1MB) - Keep in Git
- All Python scripts
- JSON config files  
- Documentation
- Small datasets

### Medium Files (1-100MB) - Consider LFS
- `dataset.jsonl` (~276 entries)
- Training logs

### Large Files (>100MB) - Exclude from Git
- Model weights (1GB+)
- Vector databases (100MB+)  
- Raw scraped data (500MB+)
- Training checkpoints

## 🛠️ Maintenance

### Regular Updates
- Add new Q&A pairs to `small_car_dataset.jsonl`
- Update dependencies in `requirements.txt`
- Refresh documentation in `README.md`

### Periodic Tasks
- Re-scrape data sources
- Re-train model with new data
- Update base model versions
- Clean up old checkpoints

## 📊 Project Metrics

### Code Organization
- **8 Python scripts** - Core functionality
- **1 Jupyter notebook** - Data processing
- **2 essential data files** - Training/inference
- **4 documentation files** - Setup/usage

### Total Project Size
- **Git repository**: ~2MB (excluding large files)
- **Full project**: ~3-5GB (with models and data)
- **Runtime memory**: 4-8GB GPU (inference)

This analysis shows a well-structured ML project with clear separation between code, data, and artifacts. The .gitignore properly excludes large generated files while preserving essential source code and documentation.
