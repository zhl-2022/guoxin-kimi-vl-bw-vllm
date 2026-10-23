# Kimi-VL-A3B-Instruct on BW + vLLM 0.18.1（session005 真实 GPU 开发验证）

## 1. 简介

本轮已在 **SCNet A Notebook `2112186448839770114`、BW 64GB ×1、BF16、TP1** 完成 Kimi-VL 的真实 native vLLM 加载及七项 HTTP 开发验证。健康、模型列表、单图、多图、两轮对话与图像 SSE 全部通过；server 有五条 chat POST200。controller 与停止入口均 exit0，最终 state 为 `stopped`。本报告作者已独立读取真实响应、配置、环境、启动和停止原件，并重算完整性与 SSE 文本；最终 A 独立清理原件已确认原两个进程身份消失、30129端口关闭。

正式精度、正式性能、官方验收和提交均未完成。合成图 smoke 证明本轮输入的服务链路，不能换算为 OCRBench/MMMU 或赛事精度。

模型固定为 `moonshotai/Kimi-VL-A3B-Instruct` revision `398eede0903cd983a2bfa0cc634e9ac1d843f375`，真实架构为 `KimiVLForConditionalGeneration`。模型约 16.4B 总参数、3B 激活；七个 BF16 权重共 **32,816,098,528 字节**，不能按 3B 常驻权重估算资源。公开支持依据为[发布方部署](https://github.com/MoonshotAI/Kimi-VL#8-deployment)及[vLLM v0.18.1 实现](https://github.com/vllm-project/vllm/blob/v0.18.1/vllm/model_executor/models/kimi_vl.py)，本报告的 BW 结论来自实际原件。

获取入口为 [SCNet 模型页](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Kimi-VL-A3B-Instruct)、[Hugging Face 固定 revision](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/tree/398eede0903cd983a2bfa0cc634e9ac1d843f375)及[固定双源清单](../tasks/kimi-vl-bw-vllm/download-sources.json)。管理克隆失败原件保留，实际使用自管目录下载并校验。完整镜像 registry/repo:tag 与本题私有备份地址尚待交付闭合，不填猜测值；赛事最终表单由用户手动提交。

## 2. 验证环境

实际环境见 [env.log](../evidence/session005/remote-kimi-final-001/kimi-tools/tasks/kimi-vl-bw-vllm/pilot-evidence/3f9dcea419780e535594e16d1df6b2b5/env.log)。该包由 B 导出共享存储中的 A 文件，不能将 B 误写成 Kimi 执行主机；[A 启动记录](../evidence/session005/remote-kimi-final-001/kimi-tools/evidence/session005/kimi-pilot-A-001-start.json)绑定 A Notebook、argv、覆盖变量及冻结 pilot SHA。

| 项 | 实际记录 |
|---|---|
| 平台 | 华中一区 A 区；BW 64GB ×1，TP1 |
| Notebook | A `2112186448839770114`；主代理观察起机14:12:29 UTC，自动关机15:00 UTC |
| Python / HIP | `3.10.12` / `6.3.26093` |
| Torch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` |
| Triton | `3.6.0+gitc73250c4.staging` |
| vLLM / vllm-hcu | `0.18.1+das.fa71803.dtk2604` / `0.18.1+das.dtk2604` |
| Transformers / tokenizers | `5.5.0` / `0.22.2` |
| 设备 | `gpu_available=true`、设备名BW、`hy-smi` exit0 |
| 模型路径 | `/root/private_data/guoxin-candidates/models/kimi-vl` |
| 工具目录 | `/root/private_data/guoxin-kimi-tools-005` |
| 端口与映射 | `127.0.0.1:30129`；无外部服务映射 |

`image_effective=null`；`/opt/dtk/.info/version` 不存在，不能把版本标签当作独立 DTK 文件验证。平台镜像选择为 DTK26.04 / vLLM Jupyter，完整镜像引用仍缺。`hy-smi` 的即时 VRAM79% 不是峰值显存或性能成绩。

[A 完整 checkpoint 校验](../evidence/session005/remote-kimi-final-001/kimi-tools/evidence/session005/kimi-checkpoint-verification-A-001.json)为 `weights_hashed=true`、`complete=true`。独立核对全部20项实际 SHA 与 [model-lock](../tasks/kimi-vl-bw-vllm/model-lock.json)一致，含七个权重，无额外可执行文件，架构与发布方一致。B 早先被 nohup 提示污染的验证输出保留；正式引用这里的干净 A JSON。

## 3. 适配优化方案

新增 [独立 runtime](../scripts/kimi_vl_runtime_005.py)与 [task](../tasks/kimi-vl-bw-vllm/task.json)，复用冻结 supervisor，不改 session004。实测 BF16、TP1、4K、单 seq、memory0.8、eager、两图、零视频；没有修改权重、结构、tokenizer、RoPE 或采样语义，没有强制 MLA/attention/MoE kernel。实际日志解析到真实 Kimi 架构，完成七个权重加载并初始化 GPU KV cache。

[upstream-source.zip](../tasks/kimi-vl-bw-vllm/upstream-source.zip)、[upstream-lock](../tasks/kimi-vl-bw-vllm/upstream-lock.json)及[静态审阅记录](../tasks/kimi-vl-bw-vllm/source-review-approved.json)固定五个发布方 Python 源码。launch 全量 hash gate 后才允许 trust_remote_code。七个实际执行脚本 SHA 已与冻结 CPU002 原字节及 export 前后 seal 三方匹配，源码 SHA 单独不作为执行证明。没有执行 pip 安装或覆盖 native 包，未运行发布方旧 CUDA/Torch/Transformers 安装脚本。发布方 [MIT 许可](../tasks/kimi-vl-bw-vllm/LICENSE_upstream)、Python 原 Apache-2.0 头部和[新编版权说明](../tasks/kimi-vl-bw-vllm/COPYRIGHT-SCOPE.txt)保留。

日志提示 slow tokenizer、旧 image_processor_class 接口弃用，以及缺 `E=64,N=1408,device_name=BW` 专门 MoE 调优表而采用默认配置。这些没有阻止本轮加载与 smoke；当前不声称完成性能调优。

独立 [CPU preflight](../scripts/kimi_cpu_preflight_005.py)不调用 AutoModel 或打开权重。001因上传文件被加6字节前缀而语法失败，原件保留；修复后完整SHA恢复到冻结版本。002/003被工具导入保护拦截，不能记为Kimi兼容失败。004已独立读取完整traceback：Transformers→TorchDynamo→native Triton的hcutuner在导入时读取GPU标签，调用torch.cuda.get_device_name并尝试初始化，因零可见设备抛出“No HIP GPUs are available”，被外层包装为AutoTokenizer导入错误。004的import/weightopen保护拒绝列表均为空，故不是socket/subprocess保护拦截，也不是缺AutoTokenizer安装包。该隐藏设备模式的CPU处理器预检未完成；前后CUDA未初始化，不能判CPU兼容通过，也不能推翻单独已验证的A GPU成功。隔离参考 [PyTorch](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)与 [AMD](https://rocm.docs.amd.com/en/latest/reference/system-optimization/gpu-isolation.html)，并以本进程实际断言约束，不影响A服务环境。

## 4. 服务启动脚本

实测外层 controller 于 **14:20:40.792096 UTC** 在 A 调用下列命令，load600/runtime1200为本轮 override，源码标准默认仍为 load900/runtime1800。

```bash
export KIMI_PYTHON_BIN=/usr/bin/python
export KIMI_MODEL_PATH=/root/private_data/guoxin-candidates/models/kimi-vl
export KIMI_LOAD_TIMEOUT=600 KIMI_MAX_RUNTIME_SECONDS=1200
timeout -s TERM -k 60s 1500s bash /root/private_data/guoxin-kimi-tools-005/tasks/kimi-vl-bw-vllm/pilot.sh
```

[实际 config](../evidence/session005/remote-kimi-final-001/kimi-tools/tasks/kimi-vl-bw-vllm/pilot-evidence/3f9dcea419780e535594e16d1df6b2b5/config.json)与 server 首行完整命令一致：

```bash
/usr/bin/python -m vllm.entrypoints.openai.api_server \
  --model /root/private_data/guoxin-candidates/models/kimi-vl \
  --tokenizer /root/private_data/guoxin-candidates/models/kimi-vl \
  --served-model-name /root/private_data/guoxin-candidates/models/kimi-vl \
  --host 127.0.0.1 --port 30129 --dtype bfloat16 \
  --tensor-parallel-size 1 --max-model-len 4096 \
  --gpu-memory-utilization 0.8 --max-num-seqs 1 \
  --max-num-batched-tokens 4096 --enforce-eager --trust-remote-code \
  --limit-mm-per-prompt '{"image":2,"video":0}'
```

运行 ID `3f9dcea419780e535594e16d1df6b2b5`；supervisor PID509/start-token279267674，server PID510/start-token279267706。state创建 **14:21:34.324392 UTC**，ready **14:23:59.678111 UTC**，stopped **14:24:41.957173 UTC**，controller结束 **14:24:43.821242 UTC**。[pilot-result](../evidence/session005/remote-kimi-final-001/kimi-tools/tasks/kimi-vl-bw-vllm/pilot-evidence/3f9dcea419780e535594e16d1df6b2b5/pilot-result.json)中controller/stop均0。

“权重加载8.15秒”只指shard阶段，不能当作外层到ready的总时间；全量hash在supervisor计时之前。停止限定本UUID，state与export前后快照字节一致。[A独立清理原件](../evidence/session005/remote-kimi-final-001/kimi-tools/evidence/session005/kimi-final-cleanup-A-001.json)于14:28:29.592782 UTC记录匹配的A hostname、同一state SHA、两个原PID的observed token均为null、30129关闭；该原件与最终导出前后state一致。

## 5. Smoke 测试

[真实 smoke](../evidence/session005/remote-kimi-final-001/kimi-tools/tasks/kimi-vl-bw-vllm/pilot-evidence/3f9dcea419780e535594e16d1df6b2b5/smoke.log)包含scope、七项case及summary。实际两张图的SHA/大小与[fixture manifest](../tasks/kimi-vl-bw-vllm/fixtures/manifest.json)一致。

| 检查 | 原件与独立核对 |
|---|---|
| health / models | 两项200；served name与配置一致 |
| 单图 | 非空回答；prompt397/completion8/total405，finish=stop |
| 多图 | 非空比较；prompt765/completion43/total808，finish=stop |
| 第一轮 | 非空回答；prompt397/completion8/total405，finish=stop |
| 追问 | 使用真实上一轮回答继续；prompt425/completion13/total438，finish=stop |
| 图像SSE | 9个未截断JSON事件，独立重拼文本相等，finish=stop，客户端DONE=true |

四条非流式response的usage加法均一致、无error、最终内容非空。SSE重拼文本为“图像中有一个红色的正方形。”，没有stream error；保留解析后的完整JSON事件和固定客户端DONE记录，没有另外保存逐字节wire数据。server恰有五条chat POST200，对应四条非流式加一条流式。

七项summary全true，开发功能链路通过。回答语义没有官方标签或评分，不当成正式精度；该结论也不覆盖任意图像、长上下文或高并发。

## 6. 精度测评

尚缺官方evaluator/selfcheck、固定subset、输入预处理/输出解析、评分实现和GPU baseline，没有正式分数、98%一致率或官方通过结论。公开OCRBench/MMMU可作为开发参考，不能冒充赛事固定评测。

[verify.py](../tasks/kimi-vl-bw-vllm/verify.py)透明调用[verify.sh](../tasks/kimi-vl-bw-vllm/verify.sh)。selfcheck只运行真实官方自检，缺材料返回2，不运行全量评测、不写accuracy PASS；未知schema保持executed_uninterpreted。本轮没有执行正式evaluator。

## 7. 性能参考

本轮仅验证4K、单seq开发输入，没有正式吞吐、TTFT、TPOT、峰值显存或加速比。日志短窗口吞吐与即时显存不填入正式性能表，五条顺序smoke不替代预热、重复、并发和尾延迟测试。

模板2K/16K × 并发1/8/64 × 两轮的12格在Kimi上均未执行。4K不能容纳16K输入+1024输出；[bench.sh](../tasks/kimi-vl-bw-vllm/bench.sh)拒绝将当前profile作为完整矩阵。后续需先验证更大上下文，再使用固定官方SGLang client、1024输出、ignoreEOS、固定图像与原生metrics。未完成档位如实披露，不填造数字，也不把缺档直接写成整题性能失败。

## 8. 注意事项与归档

冻结CPU002包ZIP SHA `27dcfad4c472690eeb80b8e1933dcd2d8c45310c6f7c8b243e3d595a04bc35ef`，manifest SHA `f7e77ae4a475b3d1aad9bab3aedb60136902f21db9f88e5302ad0d3e7d8891f3`。[bundle记录](../evidence/kimi-vl-preparation-20261023/bundle-002.json)与[CPU工程测试](../evidence/kimi-vl-preparation-20261023/validation-003/validation.json)保留准备阶段边界，新增CPUhelper不反写冻结包。

[最终独立Kimi审计](../evidence/session005/kimi-final-001-audit.json)对应B导出的共享A文件包：ZIP SHA `b48da2ab024b90bb9aeba2727252807a6ae73a399c10d62a926dafcaa8c87c96`，manifest SHA `1ff1dcea7e7480d7a2d2a60885d4f4d5e34b201c8283fb689e1812d9251bda75`。82个entry/81个payload的路径、大小、SHA、CRC和本地回读，以及35个源码前后seal均独立核对；实际七个pilot脚本与冻结包原字节一致。[原始完整性记录](../evidence/session005/remote-kimi-final-001.local-verification.json)保留。

最终包由exporter002生成，已独立确认没有checkpoint-config-only或isolated-hf-modules缓存目录entry，保留CPU001—004错误原文、修复JSON与各实际helper源SHA；不把失败预检删除或改成PASS。[初次共享包审计](../evidence/session005/kimi-pilot-shared-export-001-audit.json)作为历史保留，其重复配置metadata捕获不改写；最终pilot原件与初次审计的对应文件逐字节相同。当前最终报告只追加session005证据，不改session004。

原config中GPU_verified=false是生成时静态标记，不是运行后判分器；本报告的GPU结论来自真实架构/加载日志、native设备、匹配的state/config、五条POST与七项响应。正式字段继续false。

| 状态 | 当前值 |
|---|---|
| 完整checkpoint独立核验 | true，20项实际SHA全对 |
| GPU加载与HTTP功能独立核验 | true，7项开发检查 |
| controller / stop | 0 / 0 |
| terminal state与export前后快照 | stopped，字节一致 |
| A独立进程/端口清理证明本地核验 | true，原两token null、30129关闭 |
| CPU004本地核验 | 已核原文，预检未完成；native Triton零可见设备导入限制，不影响已核验GPU功能 |
| 正式性能负载 | 12格未运行，4K pilot范围 |
| official_accuracy_verified / official_performance_verified | false / false |
| submission_ready / contest_submission_performed | false / false |
