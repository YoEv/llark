#!/bin/bash

# Simple Hugging Face setup for LLark
set -e

echo "=== LLark Hugging Face Setup ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if huggingface-cli is installed
check_hf_cli() {
    if ! command -v huggingface-cli &> /dev/null; then
        print_info "Installing huggingface_hub..."
        pip install huggingface_hub
    fi
    print_info "✓ huggingface-cli is available"
}

# Check login status
check_login() {
    if huggingface-cli whoami &> /dev/null; then
        USER=$(huggingface-cli whoami)
        print_info "✓ Already logged in as: $USER"
        return 0
    else
        print_warning "Not logged in to Hugging Face"
        return 1
    fi
}

# Guide user through setup
guide_setup() {
    echo ""
    echo "To access Llama-2 models, you need to:"
    echo ""
    echo "1. 🔑 Get a Hugging Face token:"
    echo "   - Go to: https://huggingface.co/settings/tokens"
    echo "   - Create a new token (read access is sufficient)"
    echo "   - Copy the token"
    echo ""
    echo "2. 📝 Accept the Llama-2 license:"
    echo "   - Go to: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf"
    echo "   - Click 'Agree and access repository'"
    echo ""
    echo "3. 🔐 Login with your token:"
    echo "   - Run: huggingface-cli login"
    echo "   - Paste your token when prompted"
    echo ""
    
    read -p "Have you completed steps 1 and 2? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Great! Now let's login..."
        huggingface-cli login
    else
        print_warning "Please complete the setup steps above and run this script again."
        exit 1
    fi
}

# Test access to Llama-2
test_access() {
    print_info "Testing access to meta-llama/Llama-2-7b-chat-hf..."
    
    python3 -c "
import sys
try:
    from huggingface_hub import HfApi
    api = HfApi()
    model_info = api.model_info('meta-llama/Llama-2-7b-chat-hf')
    print('✓ Access confirmed!')
    sys.exit(0)
except Exception as e:
    print(f'✗ Access denied: {e}')
    print('Please make sure you have accepted the license at:')
    print('https://huggingface.co/meta-llama/Llama-2-7b-chat-hf')
    sys.exit(1)
"
}

# Download model files
download_files() {
    CHECKPOINT_DIR="checkpoints/meta-llama/Llama-2-7b-chat-hf/checkpoint-100000"
    
    print_info "Creating checkpoint directory: $CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR"
    
    print_info "Downloading tokenizer and config files..."
    
    python3 -c "
import os
from transformers import AutoTokenizer, AutoConfig

checkpoint_dir = '$CHECKPOINT_DIR'

try:
    print('Downloading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        use_auth_token=True
    )
    tokenizer.save_pretrained(checkpoint_dir)
    print(f'✓ Tokenizer saved to {checkpoint_dir}')
    
    print('Downloading config...')
    config = AutoConfig.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        use_auth_token=True
    )
    config.save_pretrained(checkpoint_dir)
    print(f'✓ Config saved to {checkpoint_dir}')
    
    print('')
    print('=' * 60)
    print('SUCCESS! Model files downloaded.')
    print('=' * 60)
    print(f'Location: {checkpoint_dir}')
    print('')
    print('Note: This includes only tokenizer and config files.')
    print('For actual inference, you need trained LLark weights.')
    print('=' * 60)
    
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"
}

# Main execution
main() {
    check_hf_cli
    
    if ! check_login; then
        guide_setup
    fi
    
    if test_access; then
        download_files
        print_info "Setup completed successfully!"
    else
        print_error "Setup failed. Please check the license acceptance."
        exit 1
    fi
}

# Run main function
main