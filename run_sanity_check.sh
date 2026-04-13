#!/bin/bash
# Sanity check script - Test all checkpoints on a single image
#
# Usage:
#   # Edit CONFIG and CHECKPOINT_DIR below, then:
#   ./run_sanity_check.sh
#
#   # Test with degraded input (better for showing restoration):
#   ./run_sanity_check.sh --degrade
#
#   # Or override defaults:
#   ./run_sanity_check.sh --checkpoint-dir ./experiments/2026-04-13T11-59-20_DAEFR_predegraded/

# Configuration - EDIT THESE
CHECKPOINT_DIR="./experiments/2026-04-13T11-59-20_DAEFR_predegraded/"  # Folder with checkpoints
CONFIG="configs/DAEFR.yaml"                                             # Model config
INPUT_IMAGE="./datasets/FFHQ/images512x512_validation/celeba_512_validation/00000000.png"  # Test image
OUTPUT_DIR="./sanity_check_results"                                     # Where to save results
GPU="0"                                                                 # GPU ID
DEGRADE=""                                                              # Set to "--degrade" to apply degradation

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse arguments to override defaults
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --input)
            INPUT_IMAGE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --degrade)
            DEGRADE="--degrade"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint-dir <path>   Folder containing checkpoint files"
            echo "  --config <path>           Model config file (default: configs/DAEFR.yaml)"
            echo "  --input <path>            Input test image"
            echo "  --output <path>           Output directory for results"
            echo "  --gpu <id>                GPU ID to use (default: 0)"
            echo "  --degrade                 Apply synthetic degradation to input"
            echo "  --help, -h                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}===========================================${NC}"
echo -e "${YELLOW}  DAEFR Checkpoint Sanity Check${NC}"
echo -e "${YELLOW}===========================================${NC}"

# Validate inputs
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo -e "${RED}Error: Checkpoint directory not found: $CHECKPOINT_DIR${NC}"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
    exit 1
fi

if [ ! -f "$INPUT_IMAGE" ]; then
    echo -e "${RED}Error: Input image not found: $INPUT_IMAGE${NC}"
    echo "Please provide a valid test image with --input <path>"
    exit 1
fi

echo "Configuration:"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Config: $CONFIG"
echo "  Input image: $INPUT_IMAGE"
echo "  Output dir: $OUTPUT_DIR"
echo "  GPU: $GPU"
if [ -n "$DEGRADE" ]; then
    echo "  Degradation: ENABLED (blur + noise + JPEG)"
fi
echo ""

# Run sanity check
python scripts/sanity_check_checkpoints.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config "$CONFIG" \
    --input "$INPUT_IMAGE" \
    --output "$OUTPUT_DIR" \
    --gpu "$GPU" \
    $DEGRADE

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}===========================================${NC}"
    echo -e "${GREEN}  Sanity Check Complete!${NC}"
    echo -e "${GREEN}===========================================${NC}"
    echo ""
    echo "Results in: $OUTPUT_DIR"
    echo ""
    echo "Files generated:"
    ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || echo "  (no PNG files found)"
else
    echo ""
    echo -e "${RED}===========================================${NC}"
    echo -e "${RED}  Sanity Check Failed${NC}"
    echo -e "${RED}===========================================${NC}"
    exit 1
fi
