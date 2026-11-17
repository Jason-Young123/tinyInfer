
#!/bin/bash
set -euo pipefail
shopt -s nullglob

# Parse mode (default: PPL).
MODE="PPL"
case "${1,,}" in
    ppl) MODE="PPL"; shift ;;
    top1_agreement) MODE="top1_agreement"; shift ;;
    kl_divergence) MODE="KL_divergence"; shift ;;
esac

# Select corpus files.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -gt 0 ]; then
    FILES=("$@")
else
    FILES=("$SCRIPT_DIR"/datasets/*.txt)
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No .txt files found." >&2
    exit 1
fi

# Common arguments.
COMMON_ARGS=(
    --tokenizer /home/jason/huggingface/Qwen3-4B
    --attention-backend flash
    --dtype bf16
    --max-length 4096
    --stride 512
    --chunk-size 128
    --block-size 256
)

# Run benchmarks.
for TGT in "${FILES[@]}"; do
    [ -f "$TGT" ] || { echo "File not found: $TGT" >&2; exit 1; }

    echo "========== $MODE: $(basename "$TGT") =========="

    case "$MODE" in
        PPL)
            for MODEL in Qwen3-4B Qwen3-4B-MXFP4; do
                QUANT=none
                [ "$MODEL" = "Qwen3-4B-MXFP4" ] && QUANT=mxfp4

                python -m tinyinfer.quantization.PPL \
                    --model "/home/jason/huggingface/$MODEL" \
                    --quantization "$QUANT" \
                    --text-file "$TGT" \
                    "${COMMON_ARGS[@]}"
            done
            ;;

        top1_agreement)
            python -m tinyinfer.quantization.top1_agreement \
                --model /home/jason/huggingface/Qwen3-4B \
                --quantized-model /home/jason/huggingface/Qwen3-4B-MXFP4 \
                --quantization mxfp4 \
                --text-file "$TGT" \
                "${COMMON_ARGS[@]}"
            ;;

        KL_divergence)
            python -m tinyinfer.quantization.KL_divergence \
                --model /home/jason/huggingface/Qwen3-4B \
                --quantized-model /home/jason/huggingface/Qwen3-4B-MXFP4 \
                --text-file "$TGT" \
                "${COMMON_ARGS[@]}"
            ;;
    esac
done

