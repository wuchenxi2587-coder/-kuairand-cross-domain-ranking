#!/bin/bash
# ============================================================
# 一键运行训练，灵活选择特征组合
# 用法: bash run_with_features.sh <模型> <特征模式> [其他参数]
#
# 模型选项 (渐进式架构):
#   din          - 基础DIN模型
#   din_baseline - DIN基线版本
#   psrg         - DIN + PSRG (Personalized Sequential Routing Gate)
#   pcrg         - DIN + PCRG (Personalized Candidate-aware Routing Gate)
#   transformer  - + Transformer Fusion
#   mbcnet       - + MBCNet Head (Fine-Grained Compression)
#   ppnet        - + PPNet (Parameter Personalized Network)
#
# 特征模式选项:
#   sparse       - 仅使用Sparse分桶特征 (bucket版本)
#   dense        - 仅使用Dense原始值特征
#   all          - 使用全部特征 (sparse + dense)
#
# 示例:
#   bash run_with_features.sh din all --epochs 10
#   bash run_with_features.sh transformer sparse --batch_size 256
#   bash run_with_features.sh ppnet all --epochs 5 --lr 0.001
# ============================================================

# 参数解析
MODEL=${1:-din}              # 默认din
FEATURE_MODE=${2:-all}       # 默认all
shift 2                      # 移除前两个参数，剩余传给训练脚本

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================================"
echo "KuaiRand Training with Feature Selection"
echo "============================================================"
echo "Model: $MODEL"
echo "Feature Mode: $FEATURE_MODE"
echo "============================================================"

# 根据特征模式设置环境变量（用于日志记录）
case $FEATURE_MODE in
    sparse)
        echo -e "${YELLOW}Using SPARSE features only (bucket version)${NC}"
        ;;
    dense)
        echo -e "${YELLOW}Using DENSE features only (raw values)${NC}"
        ;;
    all|both)
        echo -e "${GREEN}Using ALL features (sparse + dense)${NC}"
        ;;
    *)
        echo -e "${RED}Unknown feature mode: $FEATURE_MODE${NC}"
        echo "Usage: bash run_with_features.sh <model> <sparse|dense|all> [extra_args]"
        exit 1
        ;;
esac

# 选择配置文件
case $MODEL in
    din)
        CONFIG="configs/train_din_mem16gb.yaml"
        MODEL_NAME="DIN"
        ;;
    din_baseline)
        CONFIG="configs/din_baseline_mem16gb.yaml"
        MODEL_NAME="DIN Baseline"
        ;;
    psrg)
        CONFIG="configs/train_din_psrg_pcrg_mem16gb.yaml"
        MODEL_NAME="DIN + PSRG"
        ;;
    pcrg)
        CONFIG="configs/train_din_psrg_pcrg_mem16gb.yaml"
        MODEL_NAME="DIN + PCRG"
        ;;
    transformer)
        CONFIG="configs/train_din_psrg_pcrg_transformer.yaml"
        MODEL_NAME="DIN + PSRG/PCRG + Transformer"
        ;;
    mbcnet)
        CONFIG="configs/train_din_psrg_pcrg_transformer_mbcnet.yaml"
        MODEL_NAME="DIN + PSRG/PCRG + Transformer + MBCNet"
        ;;
    ppnet)
        CONFIG="configs/train_din_psrg_pcrg_transformer_mbcnet_ppnet.yaml"
        MODEL_NAME="DIN + PSRG/PCRG + Transformer + MBCNet + PPNet (Full)"
        ;;
    *)
        echo -e "${RED}Unknown model: $MODEL${NC}"
        echo ""
        echo "Available models:"
        echo "  din          - 基础DIN模型"
        echo "  din_baseline - DIN基线版本"
        echo "  psrg         - + PSRG"
        echo "  pcrg         - + PCRG"
        echo "  transformer  - + Transformer Fusion"
        echo "  mbcnet       - + MBCNet Head"
        echo "  ppnet        - + PPNet (完整架构)"
        exit 1
        ;;
esac

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Config file not found: $CONFIG${NC}"
    echo "Please ensure you have run the preprocessing steps first."
    exit 1
fi

echo -e "${BLUE}Config: $CONFIG${NC}"
echo -e "${BLUE}Model: $MODEL_NAME${NC}"
echo -e "${BLUE}Extra args: $@${NC}"
echo "============================================================"

# 运行训练
python -m src.main_train_din \
    --config $CONFIG \
    --feature_mode $FEATURE_MODE \
    $@

EXIT_CODE=$?

echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
else
    echo -e "${RED}Training failed with exit code: $EXIT_CODE${NC}"
fi
echo "============================================================"

exit $EXIT_CODE
