# Kimi-VL-A3B-Instruct 发布方完整验证记录（Session011）

## 1. 简介

本分支保存真实完整评测和失败结果，当前精度门槛未通过，不作为已达标项目投递。原开发源码、README、上游许可和历史文件保留。

## 2. 环境

SCNet 华中一区 A 区，BW 64 GB × 1，DTK26.04，Torch2.10.0，vLLM0.18.1 DAS。各运行实际参数、run_id 和日志以 archives 中的原始回执为准。

## 3. 适配与协议

未修改 checkpoint、标签或 tokenizer；数据集固定 revision，保留所有错误回答、超长回答与失败轮次。两项完整评测使用各自发布方可核实的基线，不混用不同 checkpoint。

## 4. 代码与复现

原 serving 代码保留在仓库原位置。`publisher011/tools` 保存本轮实际评测、启动与停止控制器、原生诊断及上游 scorer；`publisher011/archives` 保存完整响应、协议、阶段日志和性能结果，原 ZIP CRC 已核验。所有新文件由 `publisher011-manifest.json` 逐项封存。

## 5. 功能与诊断

完整 900 条请求可完成。固定 12 条原生 Transformers4.51.3/BF16/eager 对照已真实执行；七个权重分片和全部 20 个模型文件匹配发布清单。12 个原始 HTTP 请求 SHA 全部匹配，12 个 prompt token 数一致，128-token 前缀有 2 条完全相同。多数差异发生在共同前缀后；Electronics_12 原 vLLM 为连续感叹号，原生参考正常开始分析电路。此诊断不是完整准确率，不据此宣称已修复。随后相同 12 输入完成 max_num_seqs=1 的 vLLM 实测（UTC04:06:19–04:11:47），launch/stop 均返回0且服务已停止；Electronics_12 仍为连续感叹号，降低并发没有解决此现象。已保存实际安装的注意力相关实现与服务日志供离线定位，尚未归因或宣称修复。

## 6. 完整准确率

MMMU validation 全部 900 条、982 张原图。直答 512-token 协议为 49.7777778%；公开 VLMEvalKit Kimi 协议、4096-token、交错图文及最终 boxed 答案解析后为 49.2222222%。发布方 Instruct 基线 57.0%，门槛 55.0%，均未达标。旧原始评分 38.0% 和所有 900 条输出也保留，未选择性删题或改答。

来源：[固定模型卡](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/398eede0903cd983a2bfa0cc634e9ac1d843f375/README.md)、[VLMEvalKit Kimi 适配](https://github.com/open-compass/VLMEvalKit/blob/160e8a7d0cb1d4d739c1757038775cf2720d6895/vlmeval/vlm/kimi_vl.py)。

## 7. 性能

原始 ZIP 中完整保留已执行的 c1/c4、各两轮、共 96 个固定样本性能请求。性能成功不代表准确率通过；不把不同提示、不同生成预算或原生诊断当作可获得优化奖励的同负载性能对比。

## 8. 当前状态

`publisher_numeric_comparison_pass=false`，`submitted=false`。代码保存至 zhl-2022 私有分支，仅是开发与评测备份，不是最终赛事表单提交。后续若修正实现，必须重新运行声明的数据和完整评测，保留本轮不利结果。
