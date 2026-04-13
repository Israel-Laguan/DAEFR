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

while [[ $# -gt 0 ]]; do
    case $1 in
        --init)
            INIT_MODE=true
            shift
            ;;
        --checkpoint)
            CHECKPOINT_ARG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage:"
            echo "  $0 --init                    # Initialize repo with README before training"
            echo "  $0                           # Upload final checkpoint (auto-find or use FINAL_CHECKPOINT)"
            echo "  $0 --checkpoint <path>       # Upload specific checkpoint"
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
if [ -n "$CHECKPOINT_ARG" ]; then
    CHECKPOINT="$CHECKPOINT_ARG"
    echo "Using specified checkpoint: $CHECKPOINT"
elif [ -f "$FINAL_CHECKPOINT" ]; then
    CHECKPOINT="$FINAL_CHECKPOINT"
    echo "Using final checkpoint: $CHECKPOINT"
else
    # Auto-find latest checkpoint
    echo "Searching for latest checkpoint in ./experiments/..."
    
    if [ -f "./experiments/last.ckpt" ]; then
        CHECKPOINT="./experiments/last.ckpt"
    else
        CHECKPOINT=$(find ./experiments -name "*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
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
python upload_checkpoint_to_hf.py \
    --checkpoint "$CHECKPOINT" \
    --repo-id "$REPO_ID" \
    --private "$PRIVATE"

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
