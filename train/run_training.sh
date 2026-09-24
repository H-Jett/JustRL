#!/bin/bash
set -x

# =============================================================================
# JustRL Phase 3 (E1) 训练入口 —— 自动适配任意「N 机 × M 卡」拓扑
#
# 每个 pod 跑【同一份】脚本,按节点 rank 分支:
#   rank 0  -> ray head,等所有 GPU 注册齐后启动训练
#   rank >0 -> ray worker,加入 head 并常驻
#
# 训练超参一字未改(论文对齐),只把拓扑与路径换成了变量。
# 用法:
#   DRY_RUN=1 bash run_training.sh                  # 只打印探测结果,不起集群不训练
#   EXPERIMENT_NAME=xxx bash run_training.sh        # 指定实验名(见下方说明)
# =============================================================================

# --- 路径 --------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRAIN_DIR="${TRAIN_DIR:-$REPO_ROOT/train}"
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${RANK:-0}}}"   # 平台注入;日志文件名要用,先算

# 一次实验的所有产出都收在这一个目录(整棵树被 .gitignore 的 outputs 覆盖,不入库):
#   train/outputs/<EXPERIMENT_NAME>/{checkpoints,validation,tensorboard,swanlab,logs,hydra}
# 名字必须【各 pod 算出来一样】,否则两个 pod 各建一个目录(日志会分家,ckpt 更糟:
# worker 写进自己那个、head 读不到)。所以不能用 $(date +%H-%M-%S) —— 两 pod 启动差
# 几秒就不同(实测差 4s 即两个目录);只能用 JOB_NAME(提交时生成,各 pod 相同)。
EXPERIMENT_NAME="${EXPERIMENT_NAME:-JustRL-DeepSeek-1.5B-${JOB_NAME:-$(date +%Y-%m-%d_%H-%M-%S)}}"
EXP_DIR="${EXP_DIR:-$TRAIN_DIR/outputs/$EXPERIMENT_NAME}"

# --- 数据与模型 --------------------------------------------------------------
TRAIN_DATASET="${TRAIN_DATASET:-$TRAIN_DIR/data/DAPO/data/dapo-math-17k.parquet}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$REPO_ROOT/data}"        # test.parquet 在仓库根 data/ 下
TEST_DATASET="['$EVAL_DATA_DIR/AIME24/test.parquet', '$EVAL_DATA_DIR/AIME25/test.parquet', '$EVAL_DATA_DIR/AMC23/test.parquet']"
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-/volume/data/hjiang02/open_source/models/DeepSeek-R1-Distill-Qwen-1.5B}"

# --- 运维参数 ----------------------------------------------------------------
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-3}"    # 不设会攒 ~140 份 ckpt(~3.5TB)
PARALLEL_SIZE="${PARALLEL_SIZE:-1}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_WAIT_TIMEOUT="${RAY_WAIT_TIMEOUT:-1800}"             # 等 worker 注册的超时(秒)

# --- 拓扑(PyTorchJob 平台已注入,直接用)-------------------------------------
# 语义:RANK=节点序号, WORLD_SIZE=节点数, NPROC_PER_NODE=每节点卡数。
# 坑:PET_NPROC_PER_NODE 是字符串 "auto" 不能用;worker pod 的 MASTER_ADDR 是空的。
NNODES="${NNODES:-${PET_NNODES:-${NUM_WORKERS:-${WORLD_SIZE:-1}}}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}}"

# head 地址:MASTER_ADDR 为空时(worker pod)按 PyTorchJob 命名约定拼 <job>-master-0
detect_head_ip() {
    if [ -n "${HEAD_IP:-}" ]; then echo "$HEAD_IP"; return; fi
    local job="${JOB_NAME:-$(hostname | sed -E 's/-(master|worker)-[0-9]+$//')}" cand ip
    for cand in "${MASTER_ADDR:-}" "${PET_MASTER_ADDR:-}" "${job}-master-0"; do
        [ -z "$cand" ] && continue
        ip="$(getent hosts "$cand" 2>/dev/null | awk '{print $1; exit}')"
        if [ -n "$ip" ]; then echo "$ip"; return; fi
    done
    hostname -I | awk '{print $1}'                        # 单机兜底
}
HEAD_IP="$(detect_head_ip)"
echo "[topology] hostname=$(hostname) rank=$NODE_RANK nnodes=$NNODES gpus=$N_GPUS_PER_NODE head=$HEAD_IP"

# --- swanlab ----------------------------------------------------------------
# 默认 local 模式:不上传、不需要 key、不联网,直接写本地实验目录,看板用
#   swanlab watch train/outputs/<EXPERIMENT_NAME>/swanlab
# 想传云端再显式 SWANLAB_MODE=cloud(那时才读 key,local 模式下完全不碰网络)。
#
# local 模式需要 swanboard(镜像里没有),见下方 PYLIBS_DIR。
export SWANLAB_MODE="${SWANLAB_MODE:-local}"
if [ "$SWANLAB_MODE" = "cloud" ]; then
    # 【坑】set -x 会把【任何展开 $SWANLAB_API_KEY 的命令】连同 key 值打进日志,
    # 光把下面这行括起来不够 —— 判断语句本身就会被 trace 出 key。所以整段关掉
    # xtrace 做完,只留布尔量给外面用,之后不再直接引用 key。
    SWANLAB_ENV_FILE="${SWANLAB_ENV_FILE:-/volume/data/hjiang02/.secrets/swanlab.env}"
    set +x
    [ -f "$SWANLAB_ENV_FILE" ] && . "$SWANLAB_ENV_FILE"
    if [ -n "${SWANLAB_API_KEY:-}" ]; then SWANLAB_UPLOAD=1; else SWANLAB_UPLOAD=0; fi
    set -x
    [ "$SWANLAB_UPLOAD" -eq 1 ] || echo "[swanlab][WARN] mode=cloud 但没找到 key($SWANLAB_ENV_FILE 不存在且未 export SWANLAB_API_KEY),将上传失败"
else
    SWANLAB_UPLOAD=0
fi

# --- 只验证探测结果,不起集群不训练 ------------------------------------------
if [ -n "${DRY_RUN:-}" ]; then
    echo "[dry-run] 角色: $([ "$NODE_RANK" -eq 0 ] && echo 'HEAD(启动训练)' || echo 'WORKER(加入集群)')"
    echo "[dry-run] 期望集群 GPU 总数: $((NNODES * N_GPUS_PER_NODE))"
    echo "[dry-run] 实验目录: $EXP_DIR"
    echo "[dry-run] ckpt: $EXP_DIR/checkpoints"
    echo "[dry-run] swanlab: mode=$SWANLAB_MODE key=$([ "$SWANLAB_UPLOAD" -eq 1 ] && echo '已配置' || echo '不需要')"
    echo "[dry-run] 看板: swanlab watch $EXP_DIR/swanlab"
    exit 0
fi

# --- 环境变量(必须在 ray start 之前 export)---------------------------------
# 【关键】ray worker 继承的是【本节点 raylet 的环境】,不是 driver 的环境:driver 或
# runtime_env 里设 PYTHONPATH 都到不了 worker 进程。verl 未 pip 安装,所以每个 pod
# 都必须在 `ray start` 前导出 PYTHONPATH,否则远程 worker 报 No module named 'verl'。
#
# PYLIBS_DIR 放镜像里没有、又需要两个 pod 都有的包(目前只有 swanlab local 模式要的
# swanboard/peewee/ujson)。放共享盘而不是 pip 装进容器:容器 overlay 一重启就没了,
# 且 worker pod 是另一个容器、装了也看不到。只放了这三个 swanlab 专用包,不会遮蔽
# 系统里的 rich/fastapi/uvicorn,故不影响其它组件。
export PYLIBS_DIR="${PYLIBS_DIR:-/volume/data/hjiang02/pylibs}"
export PYTHONPATH="$TRAIN_DIR:$PYLIBS_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PROJECT_NAME='justrl' NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true HYDRA_FULL_ERROR=1
export CKPT_PATH="$EXP_DIR/checkpoints" \
       TENSORBOARD_DIR="$EXP_DIR/tensorboard" \
       SWANLAB_LOG_DIR="$EXP_DIR/swanlab" \
       OUTLINES_CACHE_DIR="$EXP_DIR/outlines_cache" \
       HYDRA_RUN_DIR="$EXP_DIR/hydra" \
       LOG_DIR="$EXP_DIR/logs"

# --- 运行日志 ----------------------------------------------------------------
# 同一实验目录下按 rank 分文件;落共享盘,所以 worker 那份在 master 上也能直接读。
# pod 重启会追加进同一文件(tee -a),正好保留 ray 重连过程的完整现场。
if [ "$NODE_RANK" -eq 0 ]; then
    LOG_NAME="training_rank0.log"
else
    LOG_NAME="ray_worker_rank${NODE_RANK}.log"
fi
mkdir -p "$LOG_DIR" 2>/dev/null
LOG_FILE="$LOG_DIR/$LOG_NAME"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[log] 本次运行日志 -> $LOG_FILE"

# --- Ray 集群 ----------------------------------------------------------------
wait_for_gpus() {
    local want="$1" waited=0 got=0
    while [ "$waited" -lt "$RAY_WAIT_TIMEOUT" ]; do
        got=$(python3 -c "
import ray
try:
    ray.init(address='$HEAD_IP:$RAY_PORT', logging_level='ERROR')
    print(int(ray.cluster_resources().get('GPU', 0)))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
        if [ "${got:-0}" -ge "$want" ]; then
            echo "[ray] 集群就绪: 已注册 GPU $got / 期望 $want (等待 ${waited}s)"
            return 0
        fi
        sleep 10; waited=$((waited + 10))
        [ $((waited % 60)) -eq 0 ] && echo "[ray] 等待其他节点注册... 已注册 GPU ${got:-0} / $want (已等 ${waited}s)"
    done
    echo "[ray][ERROR] 超时 ${RAY_WAIT_TIMEOUT}s: 只等到 ${got:-0} 个 GPU,期望 $want" >&2
    return 1
}

if [ "$NODE_RANK" -eq 0 ]; then
    # ================================ head ================================
    ray stop --force >/dev/null 2>&1 || true
    sleep 2
    ray start --head --port="$RAY_PORT" --num-gpus="$N_GPUS_PER_NODE" \
              --disable-usage-stats --dashboard-host=0.0.0.0
    wait_for_gpus "$((NNODES * N_GPUS_PER_NODE))" || exit 1

    export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"    # main_ppo.py 的 ray.init() 靠它定位集群
    cd "$TRAIN_DIR"
    python3 -m verl.trainer.main_ppo \
        hydra.run.dir="$HYDRA_RUN_DIR" \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        data.train_files="$TRAIN_DATASET" \
        data.val_files="$TEST_DATASET" \
        data.train_batch_size=256 \
        data.val_batch_size=6312 \
        data.max_prompt_length=1024 \
        data.max_response_length=15360 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        +data.use_boxed_suffix_prompt=True \
        actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.clip_ratio_low=0.2 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        +actor_rollout_ref.rollout.val_kwargs.max_new_tokens=31744 \
        actor_rollout_ref.rollout.val_kwargs.n=32 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        reward_model.enable=False \
        reward_model.reward_manager=dapo \
        +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
        +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
        +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
        trainer.val_before_train=True \
        "trainer.logger=['console','tensorboard','swanlab']" \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
        trainer.nnodes=$NNODES \
        trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
        trainer.save_freq=50 \
        trainer.test_freq=50 \
        trainer.total_epochs=1 \
        trainer.default_local_dir="$CKPT_PATH" \
        trainer.validation_data_dir="$EXP_DIR/validation"
    RC=$?
    echo "[head] 训练进程退出, rc=$RC"
    exit $RC
else
    # =============================== worker ===============================
    echo "[ray] worker (rank=$NODE_RANK) 加入 $HEAD_IP:$RAY_PORT"

    # `ray start --block` 连不上 head 时【不会快速失败,而是一直挂着】(实测),所以
    # 先用 TCP 探活确认 head 就绪,再交给 ray,重试才有意义。
    tcp_probe() { timeout 5 bash -c "cat < /dev/null > /dev/tcp/$HEAD_IP/$RAY_PORT" 2>/dev/null; }

    for attempt in $(seq 1 60); do
        tcp_probe && break
        if [ "$attempt" -eq 60 ]; then
            echo "[ray][ERROR] 等待 head $HEAD_IP:$RAY_PORT 超时(600s),worker 退出" >&2
            exit 1
        fi
        echo "[ray] 第 $attempt/60 次:head 尚未就绪,10s 后重试..."
        sleep 10
    done

    # 加入集群并自愈:raylet 掉线就重新加入,保证长跑期间集群不掉节点
    while true; do
        ray start --address="$HEAD_IP:$RAY_PORT" \
                  --num-gpus="$N_GPUS_PER_NODE" --disable-usage-stats --block
        tcp_probe || { echo "[ray] head 不可达,worker 结束"; break; }   # 训练已结束
        echo "[ray] worker raylet 退出,10s 后重新加入..."
        sleep 10
    done

    echo "[worker] 保持 pod 存活"          # 避免 k8s 反复重启刷噪声
    sleep infinity
fi
