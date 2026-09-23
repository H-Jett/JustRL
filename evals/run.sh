#!/bin/bash
# ===========================================================================
# JustRL 评测入口: 设置环境变量 -> 依次跑 gen_vllm.py / grade.py -> 日志落盘
#
# 建议用 nohup 后台运行, 断链不会中断:
#   nohup bash evals/run.sh >/dev/null 2>&1 &
#   tail -f outputs/logs_eval/<模型名>_<时间戳>.log
#
# 前台调试: 直接 bash evals/run.sh
# 换模型  : EVAL_MODEL=/path/to/other bash evals/run.sh
# ===========================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# 配置: 两个 python 脚本只读环境变量、不自带默认值, 默认值统一放这里
# ---------------------------------------------------------------------------
export EVAL_MODEL="${EVAL_MODEL:-/volume/data/hjiang02/open_source/models/DeepSeek-R1-Distill-Qwen-1.5B}"
export EVAL_GEN_GPUS="${EVAL_GEN_GPUS:-0,1,2,3,4,5,6,7}"   # 生成用, 物理卡号
export EVAL_GRADE_GPU="${EVAL_GRADE_GPU:-0}"       # 判分模型用, 单卡
export EVAL_DATA_DIR="${EVAL_DATA_DIR:-$REPO_ROOT/data}"
export EVAL_OUT_DIR="${EVAL_OUT_DIR:-$REPO_ROOT/outputs/justrl_eval_outputs/$(basename "$EVAL_MODEL")}"
export EVAL_VERIFIER_MODEL="${EVAL_VERIFIER_MODEL:-/volume/data/hjiang02/open_source/models/CompassVerifier-3B}"
# 判分口径: 默认 0/0 = 论文基线行的口径(纯规则 + 空题干); 详见 grade.py 同名开关
export EVAL_VERIFIER_ENABLE="${EVAL_VERIFIER_ENABLE:-0}"
export EVAL_VERIFIER_USE_QUESTION="${EVAL_VERIFIER_USE_QUESTION:-0}"

export PYTHONPATH="$REPO_ROOT/evals"
export TOKENIZERS_PARALLELISM=true

LOG="$REPO_ROOT/outputs/logs_eval/$(basename "$EVAL_MODEL")_$(TZ='Asia/Shanghai' date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")" "$EVAL_OUT_DIR"
cd "$REPO_ROOT" || exit 1

# ---------------------------------------------------------------------------
# 运行
# ---------------------------------------------------------------------------
{
    echo "==================== 配置 ===================="
    echo "EVAL_MODEL          = $EVAL_MODEL"
    echo "EVAL_GEN_GPUS       = $EVAL_GEN_GPUS"
    echo "EVAL_GRADE_GPU      = $EVAL_GRADE_GPU"
    echo "EVAL_DATA_DIR       = $EVAL_DATA_DIR"
    echo "EVAL_OUT_DIR        = $EVAL_OUT_DIR"
    echo "EVAL_VERIFIER_MODEL = $EVAL_VERIFIER_MODEL"
    echo "判分口径            = VERIFIER_ENABLE:$EVAL_VERIFIER_ENABLE USE_QUESTION:$EVAL_VERIFIER_USE_QUESTION"
    echo "=============================================="

    echo "--- [1/2] gen_vllm.py 开始 ---"
    python3 -u evals/gen_vllm.py
    echo "--- [1/2] gen_vllm.py 结束 rc=$? ---"

    echo "--- [2/2] grade.py 开始 ---"
    python3 -u evals/grade.py
    echo "--- [2/2] grade.py 结束 rc=$? ---"

    echo "==================== 完成 ===================="
} 2>&1 | tee "$LOG"
