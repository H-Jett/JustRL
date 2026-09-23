import os
import sys
import json
import random
import traceback
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

# --------------------------------------------------------------------------- #
#                   Global constants / variables                              #
# --------------------------------------------------------------------------- #
def require_env(key: str) -> str:
    """读取必填环境变量。这里刻意不设任何默认值 —— 全部配置由 evals/run.sh 提供,
    避免"代码里的默认值"和"实际机器/模型"不一致导致结果不可比。"""
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"缺少必填环境变量 {key}; 请通过 evals/run.sh 运行, 或手动 export {key}=..."
        )
    return value


DATA_DIR = require_env("EVAL_DATA_DIR")
TASKS       = [
    {"name": "AIME24", "path": f"{DATA_DIR}/AIME24/test.parquet", "N": 32},
    {"name": "AIME25", "path": f"{DATA_DIR}/AIME25/test.parquet", "N": 32},
    {"name": "AMC23", "path": f"{DATA_DIR}/AMC23/test.parquet", "N": 32},
    {"name": "MATH-500", "path": f"{DATA_DIR}/MATH-500/test.parquet", "N": 4},
    {"name": "Minerva", "path": f"{DATA_DIR}/Minerva/test.parquet", "N": 4},
    {"name": "Olympiad-Bench", "path": f"{DATA_DIR}/Olympiad-Bench/test.parquet", "N": 4},
    {"name": "BRUMO25", "path": f"{DATA_DIR}/BRUMO25/test.parquet", "N": 32},
    {"name": "CMIMC25", "path": f"{DATA_DIR}/CMIMC25/test.parquet", "N": 32},
    {"name": "HMMT25", "path": f"{DATA_DIR}/HMMT25/test.parquet", "N": 32},
]
PROMPT_TEMPLATE = """{problem} Please reason step by step, and put your final answer within \\boxed{{}}."""
MODEL       = require_env("EVAL_MODEL")
MAX_TOKENS  = 31744
TEMPERATURE = 0.7
TOP_P       = 0.9
# 每个 GPU worker 独占一段 vLLM 端口(worker 之间互不重叠), 说明见 worker_process()
PORT_BASE   = int(require_env("EVAL_PORT_BASE"))
PORT_STRIDE = 100
OUT_DIR     = Path(require_env("EVAL_OUT_DIR"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def load_samples(filepath: str):
    """Read parquet file and return a list of prompts (no duplication)."""
    df = pd.read_parquet(filepath)
    if "BRUMO25" in filepath or "CMIMC25" in filepath or "HMMT25" in filepath:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "problem"].strip(),
                "answer": df.at[i, "answer"].strip(),
            }
            for i in range(len(df))
        ]
    else:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "prompt"][0]["content"].strip(),
                "answer": df.at[i, "reward_model"]["ground_truth"].strip(),
            }
            for i in range(len(df))
        ]
    print(f"Total unique samples: {len(samples)}")
    return samples


def split_seeds(seeds: list[int], num_workers: int):
    """Round-robin split of the seed list into num_workers chunks."""
    chunks = [[] for _ in range(num_workers)]
    for idx, s in enumerate(seeds):
        chunks[idx % num_workers].append(s)
    return chunks


def save_jsonl(path: Path, items: list):
    """把生成结果按 jsonl 落盘(沿用原来的写出格式)。"""
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
#                           Worker process (one GPU)                          #
# --------------------------------------------------------------------------- #
def worker_process(args_tuple):
    """
    Each worker runs on a single GPU:

    args_tuple = (samples, seed_list, gpu_id, port_base)
    """
    samples, seed_list, gpu_id, port_base = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # vLLM 初始化时自己找空闲端口搭 rendezvous(_get_open_port 是 bind -> getsockname -> close,
    # 端口并没有被占住), 8 个 worker 同时找就可能挑到同一个 -> EADDRINUSE 整轮崩掉。
    # VLLM_PORT 是 vLLM 官方留的口子(设了就从这个端口开始往后找), 每张卡给一段独占区间即可。
    os.environ["VLLM_PORT"] = str(port_base)
    print(f"[GPU {gpu_id}] seeds={seed_list} | loading model (port base {port_base})...", flush=True)

    llm = LLM(model=MODEL, enforce_eager=True)
    results = []

    for seed in seed_list:
        sampling = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            seed=seed,
        )
        messages = [[{"role": "user", "content": s["prompt"]}] for s in samples]
        outputs = llm.chat(messages, sampling, use_tqdm=True)
        for sample, out in zip(samples, outputs):
            results.append(
                {
                    "example_id": sample["example_id"],
                    "prompt": sample["prompt"],
                    "answer": sample["answer"],
                    "seed": seed,
                    "response": out.outputs[0].text,
                }
            )
    return results


# --------------------------------------------------------------------------- #
#                                   main                                      #
# --------------------------------------------------------------------------- #
def main():
    # 物理卡号列表, 必须显式给 —— worker 内部会用它覆盖 CUDA_VISIBLE_DEVICES
    available_workers = [int(x) for x in require_env("EVAL_GEN_GPUS").split(",")]
    num_workers = len(available_workers)
    for task in TASKS:
        task_name = task["name"]
        task_path = task["path"]
        N = task["N"]

        # Update output path for the current task
        out_path = OUT_DIR / f"{task_name.lower()}_t{TEMPERATURE}_p{TOP_P}_n{N}-MNT{MAX_TOKENS}.jsonl"

        # 任一任务失败就立刻中断(非 0 退出), 由调用方整轮重跑 —— 不跳过、不留半套结果
        try:
            print(f"Starting evaluation for task: {task_name} (N={N})")

            # 1. Load original prompts
            samples = load_samples(task_path)

            # Append suffix prompt to each sample
            for sample in samples:
                sample["prompt"] = PROMPT_TEMPLATE.format(problem=sample["prompt"])

            # demo print
            print("Example prompt after formatting:")
            print(samples[0]["prompt"])

            # 2. Generate N distinct random seeds and split across GPUs
            random_seeds = random.sample(range(2**31 - 1), N)  # unique & shuffled
            seed_chunks = split_seeds(random_seeds, num_workers)

            # 3. Launch workers; 每张卡一段独占端口, 避免并发抢端口
            all_results = []
            args_list = [
                (samples, seed_chunks[i], gid, PORT_BASE + i * PORT_STRIDE)
                for (i, gid) in enumerate(available_workers)
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
                futures = [ex.submit(worker_process, tup) for tup in args_list]
                for fut in tqdm(concurrent.futures.as_completed(futures),
                                total=len(futures), desc=f"GPU workers ({task_name})"):
                    all_results.extend(fut.result())

            print(f"Total generations collected for {task_name}: {len(all_results)}")  # len(samples) * N

            # 4. Save to disk
            save_jsonl(out_path, all_results)
            print(f"Saved results for {task_name} to {out_path}")
        except Exception:
            print(f"[FAIL] {task_name} 生成失败, 中断本轮(整轮重跑):\n{traceback.format_exc()}",
                  flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()