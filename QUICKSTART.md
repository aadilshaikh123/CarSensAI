# 🚀 Quick Start Guide - Mech Q&A Bot

## ⏱️ 5-Minute Setup

### Step 1: Environment Setup
```bash
# Clone and navigate to project
cd Mech_Bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Get Hugging Face Token
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" access
3. Set environment variable:
```bash
export HF_TOKEN="your_token_here"  # Linux/Mac
set HF_TOKEN=your_token_here       # Windows
```

### Step 3: Run the App
```bash
streamlit run streamlit_app.py
```

**That's it!** 🎉 Your bot is now running at `http://localhost:8501`

---

## 🔧 Troubleshooting

### Common Issues & Fixes

**❌ "HF_TOKEN not found"**
```bash
# Solution: Set your Hugging Face token
export HF_TOKEN="hf_your_token_here"
```

**❌ "CUDA out of memory"**
```bash
# Solution: Use CPU mode or reduce batch size
# The app automatically detects and uses available hardware
```

**❌ "Model files not found"**
```bash
# Solution: App will download automatically on first run
# Ensure internet connection and HF_TOKEN are set
```

**❌ "Streamlit won't start"**
```bash
# Solution: Clear cache and restart
streamlit cache clear
streamlit run streamlit_app.py
```

---

## 🎯 Usage Tips

### Best Questions to Ask
- **Engine problems**: "What causes engine overheating?"
- **Maintenance**: "How often should I change brake fluid?"
- **Car systems**: "How does ABS work?"
- **Troubleshooting**: "Why is my car making grinding noises?"

### Features to Explore
- **Chat History**: Previous Q&A saved in session
- **Context View**: Click "Retrieved Context" to see sources
- **Real-time**: Responses generated live from the model

---

## 🚀 Next Steps

### Want to Customize?
1. **Add your own data**: Edit `small_car_dataset.jsonl`
2. **Retrain model**: Run `python finetune.py`
3. **Modify interface**: Edit `streamlit_app.py`

### Want the Full Pipeline?
1. **Collect data**: `python scrape.py`
2. **Process data**: `jupyter notebook preprocess.ipynb`
3. **Train model**: `python finetune.py`
4. **Deploy**: `streamlit run streamlit_app.py`

---

## 📚 Learn More

- **Full Documentation**: Read `README.md`
- **Technical Details**: Check `PROJECT_ANALYSIS.md`
- **Configuration**: Modify `config.ini`
- **Setup Check**: Run `python setup.py`

---

## 🆘 Need Help?

1. **Check the logs** in the terminal for error messages
2. **Verify your setup** with `python setup.py`
3. **Read the troubleshooting** section above
4. **Check file permissions** for model directories

**Happy chatting with your AI mechanic!** 🔧🤖
