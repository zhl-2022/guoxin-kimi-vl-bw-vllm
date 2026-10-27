# Kimi-VL-A3B-Instruct on BW + vLLM 0.18.1（BF16）

当前为 session016 的验证与封装版本。完整 OCRBench 已达标；完整性能矩阵、当前入口的超算复跑和固定镜像尚未全部完成，因此本版本仍标记为封装中，没有最终赛事提交或主办方验收。

本文件替代旧 session005 / session011 的“当前状态”；旧失败记录继续保留。

## 1. 简介

适配对象是 `moonshotai/Kimi-VL-A3B-Instruct`，固定 revision `398eede0903cd983a2bfa0cc634e9ac1d843f375`，使用 `KimiVLForConditionalGeneration`、视觉塔和稀疏 MoE。硬件为 SCNet 华中一区 A 区 BW 64GB 单卡，TP=1，BF16，没有量化、替换权重、改 tokenizer 或替换 checkpoint。

所选准确率范围为原版完整 OCRBench 1000 条。发布基线 867，赛事绝对容差 20 分，最低 847；当前服务配置实测 861，另一配置实测 867。MMMU 全部 900 条实测 52.3333333%，未达到 55% 门槛，继续披露，没有将其改称通过。用户授权每模型选一两个完整合适的公开数据集，本次 OCRBench 在生成任何 OCR 预测前已固定选择。

- [固定发布方模型卡](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/398eede0903cd983a2bfa0cc634e9ac1d843f375/README.md)
- [超算模型页面](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Kimi-VL-A3B-Instruct)
- [GitHub 私有代码仓库](https://github.com/zhl-2022/guoxin-kimi-vl-bw-vllm)，当前分支 `codex/ocrbench-validation016`；光合固定版本和本题镜像链接仍待本轮封装。

| 入口 | 用途 |
| --- | --- |
| `env.sh` / `run.sh` | 当前 32K、四序列、math SDPA 服务配置与受控启动 |
| `env_check.sh` / `smoke.sh` | 环境采集和单图、多图、多轮、流式功能检查 |
| `scripts/prepare_ocrbench016.py` | 从固定公开 parquet 重建全部 1000 条输入，逐文件哈希门禁 |
| `verify.py` / `verify.sh` | 完整 OCRBench 自检、真实服务评测或原始预测重新评分 |
| `bench.sh` | 原生 SGLang 0.5.12 客户端六档负载，两轮全部原件 |
| `baselines/ocrbench-publisher.json` | 发布基线、容差、失败的 MMMU 对照 |
| `publisher016/` | 完整准确率原件、独立复算和明确标注范围的失败压测诊断 |

准备数据后，按顺序运行：

```bash
python3 scripts/prepare_ocrbench016.py --output data/ocrbench
bash run.sh
bash env_check.sh
bash smoke.sh
source env.sh
python3 verify.py --selfcheck
python3 verify.py
BENCH_MODE=image bash bench.sh
```

数据准备需要 CPU 版 `pyarrow` 和 Pillow。已经准备好的数据不需要重新下载；输出目录必须不存在，防止覆盖原件。服务端依赖使用平台原生镜像，不执行会替换国产 Torch 的依赖解析安装。

## 2. 验证环境

原生栈实读版本如下；本题固定镜像地址将在镜像封装完成后加入，本版不能据此声称镜像交付完成。

| 组件 | 实读版本 |
| --- | --- |
| DTK | 26.04 原生镜像系列，原生包版本带 `dtk2604` |
| torch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` |
| vLLM | `0.18.1+das.fa71803.dtk2604` |
| vllm-hcu | `0.18.1+das.dtk2604` |
| Triton | `3.6.0+gitc73250c4.staging` |
| Transformers | `5.5.0` |
| flash-attn | `2.8.3+das.opt1.dtk2604.torch2100.20260330.g3f0061`，本配置的 Torch SDPA 不使用 flash 后端 |
| Pillow / tiktoken | `12.1.1` / `0.12.0` |
| 压测客户端 | 独立环境的原版 SGLang `0.5.12` |

实际平台插件为 `vllm_hcu`，不能把没有安装的 `sglang-das` 或 `vllm-plugin-das` 包名写成已安装。服务绑定 `127.0.0.1:30129`。检查点默认目录为 `/root/private_data/guoxin-candidates/models/kimi-vl`，启动前校验全量权重及已审阅发布方 Python 源码。

## 3. 适配优化方案

模型文件和安装目录中的原生框架代码没有改动。新增的进程内 hook 关闭 flash、memory-efficient、cuDNN SDPA，只启用 math SDPA；`VLLM_MLA_DISABLE=1`。每个实际 worker 保存后端布尔状态，不能仅凭环境变量宣称生效。

服务关闭 chunked prefill 和 prefix caching，`max_num_batched_tokens=32768`、总上下文 32768、最大四序列。完整 OCRBench 从前一配置的 867 变为 861，仍高于 847；报告采用当前 861 配置，不把前一配置分数写到新配置上。

长文本合成压测还需要客户端兼容：发布方 `KimiVLProcessor` 没有 `image_token_id` 属性，SGLang 的图文随机输入生成器会混入保留 token。客户端为已验证的 `<|media_pad|>` 映射补齐 ID，并在随机文本词表中排除会话及多模态保留 token。上游 SGLang 五个源码文件保持原字节。强制 1024 token 输出时，压测请求额外屏蔽四个输入占位字符串：`<|media_pad|>`、`<|placeholder|>`、`<|media_start|>`、`<|media_end|>`。这项请求参数变化只用于合成性能负载，准确率请求不设置它；没有缩短生成预算或删减请求数。该输入输出域约束必须与性能结果一起披露，不声称与未过滤控制符的合成输入字节相同。

早期 performance001–003 的真实内核故障和 performance004 的客户端列表错误均保留。当前 005 已完成的4K、并发 1 的两轮记录经严格独立校验有效，完整矩阵仍运行中。

## 4. 服务启动脚本

`env.sh` 加入 SSH 环境缺失的 DTK 动态库目录、ROCM_PATH 与 HIP_PATH，设置进程内 math hook。启动器保留 UUID、PID 与启动令牌，只操作匹配 UUID 的本项目进程；单项服务停止不会关闭 Notebook。

实际模型命令包含以下固定参数：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /root/private_data/guoxin-candidates/models/kimi-vl \
  --tokenizer /root/private_data/guoxin-candidates/models/kimi-vl \
  --host 127.0.0.1 --port 30129 --dtype bfloat16 \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --gpu-memory-utilization 0.8 --max-num-seqs 4 \
  --max-num-batched-tokens 32768 --enforce-eager \
  --no-enable-chunked-prefill --no-enable-prefix-caching \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":{"count":5,"width":1024,"height":1024},"video":0}'
```

准确率轮使用别名 `kimi-vl-a3b-instruct`；性能轮为兼容 SGLang 本地处理器加载而使用模型本地路径作为别名。权重、处理器和执行后端不因此改变。当前 driver 原字节 SHA256 为 `11985f3c844f03b320f674309ea19060e1bc2f1ac93619c031092d574af22dcc`。

## 5. Smoke 测试

旧归档保留健康检查、模型列表、单图、多图、多轮追问与图像 SSE 功能原件。完整 OCRBench 当前轮另完成 1000/1000 真实 HTTP 请求、零请求失败、零 length 截断。旧 smoke 属于早期 4K 配置，因此当前根入口封装完成后仍需补跑该入口的完整功能检查，不能以旧版本替代当前入口。

SSE 的终止事件、实际多图输入和模型权重身份均作为独立检查，HTTP 200 本身不能代替语义准确率。

## 6. 精度测评

数据为作者公开 `echo840/OCRBench`，revision `92a54bd1384387c178d5a07140a2d85e0a3d12e1` 的完整 test 1000 条，没有挑选、删除失败样本或缩小分母。公开 parquet SHA256 为 `d9cc9f488e5310143d8d89956c19d0bdfd71e7089025da5202643f73720df7f5`。

使用固定 VLMEvalKit `160e8a7d0cb1d4d739c1757038775cf2720d6895` 的原版 OCRBench 类评分。评分源码 SHA256 为 `27e6bc94e8a0b51d2ee401694fc7b80bcd8a36c51d324fbf4a37cce4b7c9764d`；仅以安全 literal parser 读取答案列表，评分逻辑不改。发布方 TSV 下载入口证书已失效，没有绕过 TLS；所用作者 HF 原版完整数据与该 TSV 的逐字节一致性未独立证明。

图像在请求前按发布适配器转 RGB、白底合成 alpha，并按原格式重编码。图片在前，问题在后，没有追加 MMMU 的 CoT 提示。temperature=0、top_p=1、max_tokens=4096、seed=42，四路 HTTP。答案和类别标签不发给模型。

| 完整轮次 | Text Recognition | Scene VQA | Doc VQA | KIE | HMER | 总分 | 门槛 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OCR003，前一配置 | 242 | 186 | 179 | 167 | 93 | 867 | 847 |
| OCR004，当前配置 | 237 | 185 | 179 | 167 | 93 | 861 | 847 |

原件：

- [OCR003 完整 ZIP](publisher016/session016-kimi-ocrbench003.zip)，SHA256 `8afc7d5da7720376e827cbfe84cdef6564b45dedd40ab7604f30113227b876bf`。
- [OCR004 完整 ZIP](publisher016/session016-kimi-ocrbench004.zip)，SHA256 `337547f855d98d6f212b53603d0909969a671c637288ca06d68d411ae451bd80`。
- [独立复核](publisher016/independent-audit.json)：全部 1000 输入、图片、请求哈希和原始回复核对；新可复现入口重建全部 1000 请求哈希与实测相同，再评分仍为 861。

MMMU 完整 900 条 published boxed-final 为 52.3333333%，发布基线 57%、门槛 55%，失败原件继续保留在历史目录。本报告只主张所选完整 OCRBench 达标，不扩称所有发布榜单通过。

## 7. 性能参考

正在执行原生 SGLang 0.5.12 图像客户端：4K/16K 目标随机文本、每条一张 1024×1024 随机 JPEG、1024 输出预算、并发 1/8/64；每档两轮，请求数分别 8/16/128，12 个测量组共 608 条，另有每组一条 warmup。第二轮作为参考结果，第一轮完整保存。

最先完成的 4K、并发 1、第一轮为 8/8 成功，客户端报告总输出 8192 token，输出吞吐约 14.8743 token/s。该数字只是第一轮单档，不当成完整第二轮矩阵。

原客户端 JSON 中 `request_rate=Infinity` 表示不限请求速率，汇总器因此标记 `invalid_native_result`。独立分析只允许顶层这个字段规范化为字符串 `inf`，其余 NaN/Infinity 均拒绝；逐条请求数、错误、输出长度、输入分解、有限延迟和吞吐公式另行严格核对。原始状态和原始 JSON 不修改。原生 JSONL 的输出长度可来自 SSE usage 或客户端预算回填，不能额外宣称已独立逐字节认证所有 SSE usage 与终止帧。

当前尚未取得全部 12 组有效原件，因此不填写完整吞吐汇总表，也不宣称性能交付已完成。

## 8. 注意事项

- 当前标准交付状态为封装中，未提交、未获主办方验收；缺少完整性能矩阵、当前入口超算复跑和固定镜像。
- 32K 是经过本轮完整 OCRBench 使用的服务配置，不扩称支持模型卡所有长上下文或任意多图尺寸。
- 未量化；没有对准确率数据启用压测专用输出屏蔽，也没有硬编码答案。
- 所有失败轮保留。可分享诊断包只选入日志、JSON 与源码，明确省略三个多 GB 进程 core；完整 core 与完整失败压测 ZIP 仍保存在本地及原实例，不向代码仓库发布进程内存镜像。
- 旧 `publisher011`、旧报告及旧 task pilot 字段用于追溯；当前配置以 `current_profile016`、根入口、实际 run UUID 和原件哈希为准。
- 发布方模型许可、源文件原版权及仓库 `COPYRIGHT-SCOPE.txt` 继续保留；新编代码尚未选择额外开源许可证。
