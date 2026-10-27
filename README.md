# Kimi-VL-A3B-Instruct on BW + vLLM：完整 OCRBench 与原生性能交付

本版保留完整 OCRBench、12 组原生性能报告、固定 Git 根入口复跑和独立审计。当前配置 OCRBench 为 861/1000，超过 847 门槛；原生性能共 608 条测量请求。正式登记表提交和主办方验收没有发生。镜像为下列平台私有镜像及固定tag；光合固定commit和报告附件哈希由最终提交清单记录。

本文件替代 session016 草稿的当前状态；旧失败原件、旧阶段报告和历史清单继续保留。

## 1. 简介

任务为 T02、`moonshotai/Kimi-VL-A3B-Instruct`，固定 revision `398eede0903cd983a2bfa0cc634e9ac1d843f375`。在 SCNet 华中一区 A 区 BW 64GB 单卡上，以 TP=1、BF16 运行原版视觉塔、稀疏 MoE 与 `KimiVLForConditionalGeneration`；没有量化、替换检查点、修改 tokenizer 或转换权重。

准确率采用作者公开的完整原版 OCRBench 1000 条。发布基线 867，按绝对 2 个百分点计容差，最低 847；更严格的相对 2% 下限 849.66 也通过。另测完整 MMMU 900 条为 52.3333333%，未达到 55% 下限，原失败结果继续披露。本报告只主张所选完整 OCRBench 达标。

- [固定发布方模型卡](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/398eede0903cd983a2bfa0cc634e9ac1d843f375/README.md)
- [超算模型页面](https://www.scnet.cn/ui/aihub/models/sugon_scnet/Kimi-VL-A3B-Instruct)
- [光合固定项目](https://developer.sourcefind.cn/codes/ghu_30ebea88e/guoxin-kimi-vl-bw-vllm-final-session021)，完整commit见外部提交清单。
- [GitHub 私有备份](https://github.com/zhl-2022/guoxin-kimi-vl-bw-vllm)，本版分支 `codex/submission-ready021-envimage`；以外部清单的完整 commit 为固定版本。

| 根入口 | 用途 |
| --- | --- |
| `run.sh` / `stop.sh` / `env.sh` | 原生受控服务与固定 math SDPA 配置 |
| `env_check.sh` / `smoke.sh` | 环境和单图、多图、多轮、流式功能 |
| `verify.py` / `verify.sh` | 全 1000 条准确率、自检和原件重新评分 |
| `bench.sh` | 六档、两轮、608 条原生性能；两个有界服务阶段 |
| `env.log` / `verify_result.json` / `smoke.log` | 固定031根复跑的真实原件，与完整ZIP逐字节一致 |
| `bench/round1.jsonl` / `bench/round2.jsonl` | 六组原生报告的两轮汇总，保留原始native文件和各档日志 |
| `publisher021/` | 完整性能、固定根入口、独立复算及代码核验原件 |
| `publisher016/` | 前两轮完整准确率与早期失败诊断 |

## 2. 验证环境

硬件为 BW 64GB × 1，区域华中一区 A 区；服务绑定 `127.0.0.1:30129`。DTK 26.04 原生环境，实读软件如下。

| 组件 | 实际版本 |
| --- | --- |
| torch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` |
| vLLM | `0.18.1+das.fa71803.dtk2604` |
| vllm-hcu | `0.18.1+das.dtk2604` |
| Triton | `3.6.0+gitc73250c4.staging` |
| Transformers | `5.5.0` |
| flash-attn | `2.8.3+das.opt1.dtk2604.torch2100.20260330.g3f0061` |
| Pillow / tiktoken | `12.1.1` / `0.12.0` |
| 独立压测客户端 | SGLang `0.5.12`、orjson `3.11.3`，继承原生环境其余包 |

镜像：[guoxin-kimi-vl-bw021](https://www.scnet.cn/ui/console/index.html#/image-management/detail/2115214831904727041)，固定tag `dtk2604-vllm018-kimi-final-20261028`；在华中一区A区Notebook选择该镜像版本。平台详情链接是可选用的镜像入口，不虚构公开Docker registry地址。

实际平台插件为 `vllm_hcu`。Torch SDPA 只启用 math 后端，未使用 flash SDPA。模型启动前按 `tasks/kimi-vl-bw-vllm/model-lock.json` 校验全量权重、配置与已审阅的发布方 Python 源码。原生框架安装目录没有改动。镜像内最终代码位于 `/opt/guoxin-kimi021/final-project-v3`，进入该目录后执行第四章命令；早期 `/opt/guoxin-kimi021/project` 是封装前快照，只用于追溯。

最终镜像仅封装本题的环境、完整OCR数据、独立客户端和代码，不将约32.82GB的Kimi检查点打入新镜像层。平台明确检查单层15GB，早期模型内置方案触发33.7GB拦截，已按规则改为数据盘方案；初版tag只作历史记录。数据位于 `/opt/guoxin-kimi021/data`，客户端为 `/opt/guoxin/kimi-client021/bin/python`。`/root/private_data` 是持久化挂载盘。根 `run.sh` 复用标准目录 `/root/private_data/guoxin-candidates/models/kimi-vl` 的文件，再由原生启动器执行完整SHA256校验；目录缺失或文件不完整时，先调用固定ModelScope下载器取得同一revision的20个精确文件，逐个核对不可变URL、大小和SHA256，之后才启动GPU服务。已有不同终态文件会被拒绝覆盖，不使用软链接、不换检查点。

## 3. 适配优化方案

新增进程内 hook 关闭 flash、memory-efficient、cuDNN SDPA，只启用 math SDPA，并设置 `VLLM_MLA_DISABLE=1`。每个实际 worker 保存后端布尔状态，不能仅凭环境变量声称生效。关闭 chunked prefill 和 prefix caching，`max_num_batched_tokens=32768`、上下文 32768、最大服务序列 4。

发布方 `KimiVLProcessor` 缺少 SGLang 图文输入生成器需要的 `image_token_id`；客户端 hook 为已验证的 `<|media_pad|>` 补齐映射，并从合成随机文本词表排除保留 token。原版 SGLang 五个相关源码文件保留原字节。合成性能请求还屏蔽 `<|media_pad|>`、`<|placeholder|>`、`<|media_start|>`、`<|media_end|>` 四个输入占位字符串，防止进入缺少输入资产的多模态分支；输出预算仍为 1024。真实 OCR 不使用这项屏蔽。性能结果不能扩称为未过滤控制符输入的原字节结果。

旧单轮 5400 秒不足以跑完 16K/c64，实际完整两轮墙钟分别为 6026.849716、5981.359823 秒。因此本版根 `bench.sh` 使用两个独立服务阶段：前五档总预算 15000 秒，最后一档两轮总预算 19000 秒；每组上限 9000 秒，每个服务的原有 21600 秒看门狗保持不变。每阶段使用新 UUID、独立输出目录和所有权检查，不修改运行中源码、时钟或平台限制。

新版前端的全部 12 条原生客户端命令已与真实测量 argv 比较，仅规范化解释器及输出路径；覆盖全部 608 条，五类错误所有权被拒绝，保留 venv 解释器路径。新审计器在真实 PF006 两份原件上复核 256 条通过。新增前端本身只做 CPU 控制验证，没有为更新编排器重复采样全部 608 条，也不把该 CPU 验证称为新的 GPU 测量。

## 4. 服务启动脚本

镜像内进入 `/opt/guoxin-kimi021/final-project-v3` 后执行以下顺序。脚本均从仓库根目录运行；同账号数据盘已有检查点时直接复用，新账号缺失时先下载固定文件。离线环境须预先提供同一完整检查点，不会回退到其他模型。

```bash
source env.sh
bash run.sh
bash env_check.sh
bash smoke.sh
python3 verify.py --selfcheck
python3 verify.py --output verify-full-unique
bash bench.sh --plan-only
bash bench.sh
bash stop.sh
```

`bench.sh` 会确认并停止同项目、同配置的旧服务，再顺序运行两个受控服务阶段；遇到不匹配的 UUID、服务配置或被占用端口会拒绝继续。所有阶段仅停止自身模型服务，不关闭 Notebook。准确率默认读取 `KIMI_DATA_DIR`；根 `env.sh` 在本地数据不存在、镜像预置数据存在时选择 `/opt/guoxin-kimi021/data`。

镜像外的准备命令：

```bash
python3 scripts/download_kimi_vl_005.py download --provider modelscope --max-seconds 6900
python3 scripts/prepare_ocrbench016.py --output data/ocrbench
python3 scripts/prepare_bench_client021.py --output .venv-client
```

数据准备需要 CPU 版 pyarrow 与 Pillow。客户端安装只对新 venv 执行固定两包的 `--no-deps` 安装，不替换国产 Torch。下载脚本的固定模型 revision、目标目录和哈希门禁由其帮助与 model-lock 指定。准备目录须不存在，避免覆盖原件。

实际模型参数为 BF16、TP=1、max-model-len 32768、max-num-seqs 4、max-num-batched-tokens 32768、gpu-memory-utilization 0.8、enforce-eager、禁用 chunked-prefill 与 prefix-cache；每请求最多五张限定尺寸图像、video=0。服务 driver SHA256 `11985f3c844f03b320f674309ea19060e1bc2f1ac93619c031092d574af22dcc`。根复跑和性能均使用标准模型路径作为 served name。

## 5. Smoke 测试

2026-10-27 21:53–22:03 UTC，从固定 Git `031051314c2a79ee76a9fbd57d363ba11244719b` 的原始 ZIP 新建项目目录，实际执行根 run、环境采集、smoke、自检、全 1000 条准确率和自身 stop，六阶段均退出 0。原件为 `publisher021/session021-kimi-portable-replay002.zip`，不能用旧 4K 配置 smoke 替代本轮。

本轮 smoke 覆盖健康与模型身份、单图、多图、多轮及图像 SSE。完整 OCR1000 另有 1000 个唯一输入 ID、1000 个唯一原始响应 ID、1000 个重建请求哈希与图片哈希全部相符；全部 `finish_reason=stop`，零失败、零 length 截断。原始回复文本与评分文本逐条一致，重新评分仍为 861。

封装版相对实测固定入口只新增数据盘检查点复用及固定文件下载、数据目录回退、实际性能客户端 hook 和两阶段性能编排。检查点准备的缺失文件计划及固定20文件来源已做CPU验证，没有把另一次完整权重下载或新GPU复跑写成已完成。模型 driver、math hook、准确率 evaluator、数据和评分器保持实测原字节。封装前端的验证范围见第三章；尚未把另一台全新 Notebook 的整套新镜像复跑写成已完成。

## 6. 精度测评

作者公开 `echo840/OCRBench`，固定 revision `92a54bd1384387c178d5a07140a2d85e0a3d12e1`，完整 test1000，无筛选或缩减分母。parquet SHA256 `d9cc9f488e5310143d8d89956c19d0bdfd71e7089025da5202643f73720df7f5`。固定 VLMEvalKit `160e8a7d0cb1d4d739c1757038775cf2720d6895` 的 OCRBench 类，评分源码 SHA256 `27e6bc94e8a0b51d2ee401694fc7b80bcd8a36c51d324fbf4a37cce4b7c9764d`；只用安全 literal parser 读取答案列表，评分逻辑不变。

图片转 RGB、alpha 白底合成并按原格式重编码；图像在前、问题在后，不追加 MMMU CoT。temperature=0、top_p=1、max_tokens=4096、seed=42、HTTP 并发 4。答案和类别不发给模型；完整 1000 请求已独立重建哈希验证。

| 完整轮次 | Text Recognition | Scene VQA | Doc VQA | KIE | HMER | 总分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OCR003，前一配置 | 242 | 186 | 179 | 167 | 93 | 867 |
| OCR004，当前配置 | 237 | 185 | 179 | 167 | 93 | 861 |
| 固定 Git 根入口 replay002，当前配置 | 237 | 186 | 179 | 167 | 92 | 861 |

当前根复跑距绝对下限 847 为 14 分，距更严格相对下限 849.66 为 11.34 分；不需要为了达标再挑样本。前两轮原件保留在 `publisher016/`，新根复跑、独立请求复核和重新评分位于 `publisher021/`。作者 HF 原版完整数据与已失效 TSV 下载入口的逐字节一致性未独立证明，没有绕过 TLS。

MMMU900 的 52.3333333% 低于发布基线 57% 对应下限 55%，不改称通过。性能早期内核故障也不混入准确率分数。

## 7. 性能参考

原生 SGLang 0.5.12 OpenAI chat 图像客户端：目标随机文本 4096/16384 token，每请求一张 1024×1024 随机 JPEG，输出预算 1024，客户端并发 1/8/64；每档两轮分别 8/16/128 请求，共 12 组、608 条测量请求。每组另有预热，不混入 608 计数。第一轮原件完整保存，下面报告第二轮，与根 `bench/round2.jsonl` 完全一致；根 `bench/native/` 和各档日志为原字节。新压测默认在 `bench/时间戳-UUID/` 写新结果，不覆盖提交原件。

| 文本目标 | 并发 | 测量请求 | 测量秒数 | 输出 token/秒 | 平均 TTFT 毫秒 | P99 TTFT 毫秒 | 平均 TPOT 毫秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 8 | 547.260147 | 14.969115 | 1365.247110 | 1485.645071 | 65.533483 |
| 4096 | 8 | 16 | 291.540954 | 56.197936 | 59002.414229 | 77682.600248 | 67.306329 |
| 4096 | 64 | 128 | 2385.525714 | 54.944702 | 844751.673581 | 1126057.690795 | 67.722188 |
| 16384 | 1 | 8 | 629.757865 | 13.008174 | 11300.144247 | 11607.024805 | 65.902410 |
| 16384 | 8 | 16 | 719.165922 | 22.781947 | 221851.957758 | 292419.489920 | 70.898863 |
| 16384 | 64 | 128 | 5821.352308 | 22.515731 | 2136277.875056 | 2846980.419100 | 72.063344 |

客户端并发较高时，四序列服务会排队；TTFT 含排队时间。文本目标长度与分词往返后的真实计数不同，视觉 token 为每请求 1093，完整输入分解与 12 个原生文件 SHA 见 `publisher021/performance-summary.md` 和联合审计 JSON。

原 PF005 前十份完整原件加新 PF006 两份构成唯一完整矩阵，模型、driver、hook、实际服务配置逐键一致，仅 task 路径和 UUID 不同；中断和未启动组保留原状态，没有重采样已完成组。远端与本地独立审计的全部指标、源代码哈希及原生文件 SHA 一致。

原 JSON 的 `request_rate=Infinity` 表示不限速；原生产者因此报格式错误。独立审计只允许顶层该字段规范化为字符串 `inf`，拒绝其他非有限数，逐条核对请求数、状态、数组和公式，原始 JSON 不改写。原生输出长度可能来自 SSE usage 或客户端预算回填；本报告不额外声称已独立逐字节认证所有 wire token 与终止帧。

Prefix cache 关闭，多模态 processor cache 保留框架默认启用，第二轮日志显示约 50.4% MM cache 命中。不能把 prefix-cache 关闭写成所有缓存关闭。早期 001–003 内核故障、004 客户端错误及 005 未完成的两组继续保留，不计入有效矩阵。

## 8. 注意事项

- 数值达标、根入口实测、材料封装、镜像新实例复现、主办方验收与表单提交分别记录。`submitted=false` 只有真实成功页面或提交记录才能改变。
- 本仓库使用已验证 32K、四序列配置，不扩称支持全部发布方长上下文或任意图像输入；合成性能专用保留 token 策略必须随数据披露。
- 保存镜像前核对代码、完整数据、客户端与实际层大小；本题检查点存数据盘或按固定来源下载。平台15GB门禁通过后才执行保存，外部提交清单保存真实镜像ID、固定代码版本、最终八章报告与哈希，不把创建中写成已成功。
- 旧清单如 `publisher016-manifest.json` 只对应其历史生成时点。当前交付以 `publisher-delivery-manifest021.json` 的范围、固定提交、原件 ZIP 和真实运行记录为准。
- 失败诊断包明确省略多 GB core；原实例及本地仍保留完整原始失败归档，不向仓库上传进程内存镜像。
- 仓库、镜像默认私有，跨账号评审访问尚未独立验证；按赛事实际要求提供相应可访问链接，不擅自公开账号资产。
- 发布方许可、上游源文件版权及 `COPYRIGHT-SCOPE.txt` 保留，新编代码暂未选择额外开源许可证。
