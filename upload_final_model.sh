#!/bin/bash
# Upload final DAEFR checkpoint to Hugging Face after training completes
#
# Usage:
#   # Step 1: Initialize repo before training (creates README)
#   ./upload_final_model.sh --init
#
#   # Step 2: Upload final checkpoint after training
#   ./upload_final_model.sh
#
#   # Upload all checkpoints from folder, preserving structure
#   ./upload_final_model.sh --upload-all
#
# Or in one command after training:
#   ./upload_final_model.sh --checkpoint ./experiments/DAEFR_model.ckpt

# Configuration - EDIT THESE
REPO_ID="your-username/DAEFR-final"  # Change to your HF username and desired model name
PRIVATE=false                         # Set to true for private repo
FINAL_CHECKPOINT="./experiments/DAEFR_model.ckpt"  # Expected final checkpoint path
EPOCHS=100                            # Training epochs (for README)
# Get token from environment or paste here (not recommended for security)
HF_TOKEN="${HF_TOKEN:-}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
INIT_MODE=false
CHECKPOINT_ARG=""
USE_BEST_EPOCH=true  # Default: find highest epoch checkpoint (e.g., epoch=000048)
UPLOAD_ALL=false   # Upload all checkpoints from folder
PRESERVE_STRUCTURE="--preserve-structure"  # Default: preserve folder structure

while [[ $# -gt 0 ]]; do
    case $1 in
        --init)
            INIT_MODE=true
            shift
            ;;
        --checkpoint)
            CHECKPOINT_ARG="$2"
            USE_BEST_EPOCH=false
            shift 2
            ;;
        --use-best-epoch)
            USE_BEST_EPOCH=true
            shift
            ;;
        --use-latest-time)
            USE_BEST_EPOCH=false
            shift
            ;;
        --upload-all)
            UPLOAD_ALL=true
            USE_BEST_EPOCH=false
            shift
            ;;
        --flat)
            PRESERVE_STRUCTURE=""
            shift
            ;;
        --help|-h)
            echo "Usage:"
            echo "  $0 --init                    # Initialize repo with README before training"
            echo "  $0                           # Upload checkpoint with highest epoch number (default)"
            echo "  $0 --checkpoint <path>       # Upload specific checkpoint"
            echo "  $0 --use-latest-time         # Find by modification time instead of epoch"
            echo "  $0 --upload-all              # Upload ALL checkpoints from folder (preserves structure)"
            echo "  $0 --flat                    # Upload flat (no subfolders, overrides --upload-all structure)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}===========================================${NC}"
echo -e "${YELLOW}  DAEFR Model Upload Script${NC}"
echo -e "${YELLOW}===========================================${NC}"

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

# Set token
export HF_TOKEN="$HF_TOKEN"

# Step 1: Initialize repo with README (do this before training)
if [ "$INIT_MODE" = true ]; then
    echo "Initializing repository with README..."
    python upload_checkpoint_to_hf.py \
        --init \
        --repo-id "$REPO_ID" \
        --private "$PRIVATE" \
        --epochs "$EPOCHS" \
        --final-checkpoint "$FINAL_CHECKPOINT"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}===========================================${NC}"
        echo -e "${GREEN}  Repository Initialized!${NC}"
        echo -e "${GREEN}===========================================${NC}"
        echo "Repo URL: https://huggingface.co/$REPO_ID"
        echo ""
        echo "After training completes, run this script again without --init:"
        echo "  ./upload_final_model.sh"
    fi
    exit $?
fi

# Step 2: Upload final checkpoint (auto-find or use specified)
if [ "$UPLOAD_ALL" = true ]; then
    echo "Uploading ALL checkpoints from $CHECKPOINT_DIR..."
    echo "This will preserve folder structure in HF repo."
    echo ""
    
    # Run upload-all with preserved structure
    python upload_checkpoint_to_hf.py \
        --upload-all \
        --experiments-dir "$CHECKPOINT_DIR" \
        --repo-id "$REPO_ID" \
        --private "$PRIVATE" \
        $PRESERVE_STRUCTURE
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}===========================================${NC}"
        echo -e "${GREEN}  All Checkpoints Uploaded!${NC}"
        echo -e "${GREEN}===========================================${NC}"
        echo "Model URL: https://huggingface.co/$REPO_ID"
        echo ""
        echo "Files uploaded to folder: $(basename "$CHECKPOINT_DIR")/"
    else
        echo ""
        echo "Upload failed. Check errors above."
        exit 1
    fi
    exit 0
fi

if [ -n "$CHECKPOINT_ARG" ]; then
    CHECKPOINT="$CHECKPOINT_ARG"
    echo "Using specified checkpoint: $CHECKPOINT"
elif [ "$USE_BEST_EPOCH" = true ]; then
    # Find highest epoch checkpoint (e.g., epoch=000048)
    echo "Searching for highest epoch checkpoint in ./experiments/..."
    
    # Run Python to find best epoch
    CHECKPOINT=$(python -c "
import sys
sys.path.insert(0, '.')
from upload_checkpoint_to_hf import find_best_epoch_checkpoint
ckpt = find_best_epoch_checkpoint('./experiments')
print(ckpt if ckpt else '')
" 2>/dev/null)
    
    if [ -n "$CHECKPOINT" ]; then
        echo "Found: $CHECKPOINT"
    else
        echo "No epoch checkpoint found, falling back to time-based search..."
        USE_BEST_EPOCH=false
    fi
fi

# If best epoch didn't find anything, fall back to time-based or FINAL_CHECKPOINT
if [ -z "$CHECKPOINT" ] && [ -z "$CHECKPOINT_ARG" ]; then
    if [ -f "$FINAL_CHECKPOINT" ]; then
        CHECKPOINT="$FINAL_CHECKPOINT"
        echo "Using final checkpoint: $CHECKPOINT"
    else
        # Auto-find latest checkpoint by time
        echo "Searching for latest checkpoint by time in ./experiments/..."
        
        if [ -f "./experiments/last.ckpt" ]; then
            CHECKPOINT="./experiments/last.ckpt"
        else
            CHECKPOINT=$(find ./experiments -name "*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        fi
    fi
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found!"
    echo "Make sure training has completed or specify with --checkpoint <path>"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo -e "${GREEN}Found checkpoint: $CHECKPOINT${NC}"
echo "File size: $(du -h "$CHECKPOINT" | cut -f1)"

# Verify checkpoint is valid (not empty/corrupted)
if [ ! -s "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file is empty!"
    exit 1
fi

echo ""
echo -e "${YELLOW}Upload Configuration:${NC}"
echo "  Repository: $REPO_ID"
echo "  Private: $PRIVATE"
echo "  Checkpoint: $CHECKPOINT"
echo ""

# Confirm upload
read -p "Proceed with upload? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 0
fi

# Run upload
if [ "$USE_BEST_EPOCH" = true ] && [ -z "$CHECKPOINT_ARG" ]; then
    python upload_checkpoint_to_hf.py \
        --use-best-epoch \
        --repo-id "$REPO_ID" \
        --private "$PRIVATE"
else
    python upload_checkpoint_to_hf.py \
        --checkpoint "$CHECKPOINT" \
        --repo-id "$REPO_ID" \
        --private "$PRIVATE"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}===========================================${NC}"
    echo -e "${GREEN}  Upload Complete!${NC}"
    echo -e "${GREEN}===========================================${NC}"
    echo "Model URL: https://huggingface.co/$REPO_ID"
else
    echo ""
    echo "Upload failed. Check errors above."
    exit 1
fi
