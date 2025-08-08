#!/usr/bin/env python3
"""
Setup script for Mech Q&A Bot
This script helps set up the environment and download necessary models.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Please upgrade your Python installation.")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_cuda():
    """Check if CUDA is available."""
    print("\n🔍 Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available! Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available. Fine-tuning will be slow on CPU.")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet. CUDA check will be done after installation.")
        return False

def install_requirements():
    """Install Python dependencies."""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def check_hf_token():
    """Check if Hugging Face token is set."""
    print("\n🔑 Checking Hugging Face token...")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("✅ HF_TOKEN environment variable is set")
        return True
    else:
        print("⚠️  HF_TOKEN environment variable not found")
        print("   You'll need this for model access. Set it with:")
        print("   Linux/Mac: export HF_TOKEN='your_token_here'")
        print("   Windows: set HF_TOKEN=your_token_here")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        return False

def check_model_files():
    """Check if model files exist."""
    print("\n📁 Checking model files...")
    
    base_model_path = Path("llama-3.2-1b-base")
    adapter_path = Path("llama-3.2-1b-instruct-cars-finetuned-adapter")
    
    base_exists = base_model_path.exists() and (base_model_path / "config.json").exists()
    adapter_exists = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()
    
    if base_exists:
        print("✅ Base model found")
    else:
        print("⚠️  Base model not found. Run 'python l.py' to download it.")
    
    if adapter_exists:
        print("✅ Fine-tuned adapter found")
    else:
        print("⚠️  Fine-tuned adapter not found. Run 'python finetune.py' after setup.")
    
    return base_exists, adapter_exists

def check_data_files():
    """Check if data files exist."""
    print("\n📊 Checking data files...")
    
    dataset_files = [
        "small_car_dataset.jsonl",
        "dataset.jsonl"
    ]
    
    all_exist = True
    for file in dataset_files:
        if Path(file).exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found!")
            all_exist = False
    
    return all_exist

def main():
    """Main setup function."""
    print("🚗 Mech Q&A Bot Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check CUDA (optional)
    cuda_available = check_cuda()
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    if not install_requirements():
        print("❌ Failed to install dependencies. Please check the error messages above.")
        sys.exit(1)
    
    # Re-check CUDA after PyTorch installation
    if not cuda_available:
        check_cuda()
    
    # Check HF token
    hf_token_set = check_hf_token()
    
    # Check model files
    base_exists, adapter_exists = check_model_files()
    
    # Check data files
    data_exists = check_data_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print("=" * 50)
    
    if not data_exists:
        print("❌ Missing data files! Please ensure dataset files are present.")
        return False
    
    print("✅ All dependencies installed successfully!")
    
    if hf_token_set and base_exists:
        print("✅ Ready to run Streamlit app: streamlit run streamlit_app.py")
    elif not hf_token_set:
        print("⚠️  Set HF_TOKEN before downloading models")
    elif not base_exists:
        print("⚠️  Download base model first: python l.py")
    
    if not adapter_exists:
        print("ℹ️  Optional: Train your own model with: python finetune.py")
    
    print("\n🎉 Setup complete! Check the README.md for detailed usage instructions.")
    return True

if __name__ == "__main__":
    main()
