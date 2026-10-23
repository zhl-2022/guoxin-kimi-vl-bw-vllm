# Kimi-VL-A3B-Instruct BW/vLLM 准备与有界 pilot

本任务已完成 CPU 工程准备，尚未启动 GPU、安装依赖或取得正式成绩。模型固定为 `moonshotai/Kimi-VL-A3B-Instruct` revision `398eede0903cd983a2bfa0cc634e9ac1d843f375`，真实架构为 `KimiVLForConditionalGeneration`；不修改架构或猜测强制 MLA/attention kernel。新代码保留版权范围说明；发布方 MIT 许可和 Python 文件中的 Apache-2.0 原声明保留。

## 1. 输入与保护边界

- `model-lock.json` 固定 20 个必要文件，含七个 BF16 权重及各文件完整 SHA256/大小，权重共 32,816,098,528 字节。未在本机下载权重。
- `download-sources.json` 固定 Hugging Face revision 与 ModelScope 每文件不可变 commit；20 项 SHA/大小相同。下载器最终再校验磁盘字节，不信 `main`、克隆进度或同名文件。
- `upstream-source.zip`、`upstream-lock.json` 保存五个原始 Python 文件和配置/模板，`source-review-approved.json` 记录静态审阅。静态审阅不证明 Transformers 5.5、DAS 或 GPU 兼容；没有安装旧 Transformers/FlashAttention。
- 默认 TP1、BF16、4K 上下文、`max_num_seqs=1`、eager、最多两图、禁视频，服务守护默认 1800 秒，允许上界 6900 秒。load timeout 默认 900 秒。平台整实例自动关机与前序 Trident 用时仍由主流程控制。
- `run.sh` 真启动前对全部文件做完整 hash，核五个 Python 与已审阅集合、真实架构、拒绝额外代码/权重。进程控制复用原有 stdlib supervisor，但本任务目录、端口、run ID 与配置隔离，公共源码不改动。

## 2. 优先复用 SCNet 文件，失败则双源续传

从上传包根目录执行，使用已验证的 native 解释器；以下路径是本轮主代理提供的预期映射，先实际 `ls/realpath` 核对。平台克隆最后一次由主代理观察为失败，不能写成已完成。

```bash
export KIMI_PYTHON_BIN=/usr/bin/python
KIMI_CLONE=/root/private_data/SothisAI/model/Aihub/Kimi-VL-A3B-Instruct/main/Kimi-VL-A3B-Instruct

# 只读：如果 clone 完整且通过全部 hash，可以直接 KIMI_MODEL_PATH 指向它运行。
"$KIMI_PYTHON_BIN" -B scripts/download_kimi_vl_005.py verify --model-dir "$KIMI_CLONE" > kimi-clone-verification.json
```

若 clone 不完整或失败，可先只拷贝其中完整且 hash 正确的文件到自管目录，不改管理目录；然后补齐缺项。下列是备选分支，不要求先运行全部命令。

```bash
"$KIMI_PYTHON_BIN" -B scripts/download_kimi_vl_005.py copy-existing --source "$KIMI_CLONE" --reuse-verified-files-only --max-seconds 600
"$KIMI_PYTHON_BIN" -B scripts/download_kimi_vl_005.py download --provider modelscope --max-seconds 1800 > kimi-download-modelscope.log 2>&1
# 只有上一来源失败且实例剩余预算允许，才重跑另一来源；相同身份的 .part 可以续传。
"$KIMI_PYTHON_BIN" -B scripts/download_kimi_vl_005.py download --provider huggingface --max-seconds 1800 > kimi-download-huggingface.log 2>&1
```

自管目标固定在 `/root/private_data/guoxin-candidates/models/kimi-vl`，其他候选目录不可写。已有 final 必须 hash 一致才复用，错误文件拒绝覆盖；owned partial 保留身份，Range/大小/磁盘 SHA 校验后发布。锁与每次 transfer 终态保存在 `guoxin-candidates`。下载 deadline 到期保留部分文件，不能把未完成下载写成模型失败。

## 3. 真正 pilot 与停止

先停止/确认前序 Trident 自有服务释放设备，再选择一个**已经完整通过 hash**的本地模型路径。这里的命令是待主流程执行的操作说明，本次 CPU 准备没有运行它。

```bash
export KIMI_MODEL_PATH=/root/private_data/guoxin-candidates/models/kimi-vl
export KIMI_MAX_RUNTIME_SECONDS=1800 KIMI_LOAD_TIMEOUT=900
bash tasks/kimi-vl-bw-vllm/run.sh --dry-run
bash tasks/kimi-vl-bw-vllm/pilot.sh
```

`pilot.sh` 创建唯一输出目录，保存启动/环境/功能/停止原文和运行源码 SHA，默认 smoke 输出最多 128 token；退出时只对本 run ID 调用 stop，并保存状态、配置和服务日志。失败也归档，不能当成功。单图/多图/多轮/SSE 的实际回答与完整性由通用开发 smoke 检查，不产生 OCRBench/MMMU 精度分数。需要单独停止时：

```bash
bash tasks/kimi-vl-bw-vllm/stop.sh --run-id <actual-32hex-run-id>
```

`env_check.sh --dry-run` 仅 distribution 元数据，普通 `env_check.sh` 才采集 native 设备/环境。`verify.py --selfcheck` 透明调用真实官方桥，缺官方脚本返回 2，不跑全量评测、不生成精度 PASS；未知外部 schema 保持 `executed_uninterpreted`。默认 4K pilot 无法容纳模板 16K+1K 档，`bench.sh` 明确返回 2；更大配置须另行 GPU 验证后再压测。

## 4. 当前验证与正式缺项

CPU 测试覆盖配置/时间上界、任务隔离、双源锁、坏文件/额外代码/架构拒绝、静态 source-review gate、dry-stop 不发信号、失败 clone 仅复用正确文件、实际 Bash dry-run 与缺材料 selfcheck。CPU 假权重 fixture 仅验证校验逻辑，绝不表示模型/GPU 通过。验证原件在 `evidence/kimi-vl-preparation-20261023/`。

仍待实际 native processor/tokenizer import、BW MLA/MoE/MoonViT、模型启动与 HTTP smoke、正式分题 evaluator/data/GPU baseline、适用性能和完整镜像复现。公开支持依据是[发布方 vLLM 部署](https://github.com/MoonshotAI/Kimi-VL#8-deployment)与[vLLM v0.18.1 实现](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/models/kimi_vl.py)。当前未取得固定 HYGON Kimi-VL BW recipe 证明；所选小负载是本轮开发控制参数，不伪称官方 BW 成绩。
