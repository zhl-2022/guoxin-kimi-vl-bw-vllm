# Session021 Kimi 有界接续与活动队列

更新时间：2026-10-27 18:33 UTC。该文件取代 session019 文档中 Kimi PID38105、43060 的活动状态；旧原件和原始日志保留。

## 当前结论

ASR、Trident 两项技术交付完整，仍为零项正式提交。登记表在本轮刷新后仍显示“暂不支持填写”，八卡申请仍“待开通”，实际继续使用两张既有 BW 卡。两个 Notebook 均保留。

Kimi 完整 OCRBench 两轮分数867、861已通过847门槛；完整 MMMU900 的52.3333%仍未通过55，继续披露。本轮只修正剩余性能测试的作业墙钟时间预算，不更改模型、数据、每条1024输出token预算、采样或服务调度配置。

## 本轮接续依据与已执行动作

原 PF005 的16K输入、64并发第一轮在66/128时已耗时50:02，按稳定完成速度约需98分钟，超过原单轮5400秒（90分钟）预算。2026-10-27 18:29:17 UTC向原客户端封装器发送SIGINT，让它按自身清理逻辑回收原生客户端进程组。原服务随后自行停止，旧父进程和客户端均已退出；没有修改运行中源码、时钟、看门狗或平台限制。

原10份完整原生报告、352条测量请求经远端及本地严格独立审计再次核验。未完成的16K/c64第一轮保留 `interrupted`、退出130、墙钟3485.93223秒；第二轮保留 `not_attempted_interrupted`。原 `run.json`、原始结果及日志均未改写成通过。旧根入口等待进程43060主动取消，原记录中的 `KeyboardInterrupt` 是预期控制流。

完整旧运行归档为 `dist/session016-kimi-performance005.zip`，3,827,173字节，SHA256 `5fa295e81c3d6f4af0af3e42f8d005575c634a5ba65065c5bc0259862ae83d38`。CRC与238个清单文件全部核对；本地独立审计位于 `evidence/session021/pf005-local-strict-analysis`。审计程序返回1表示完整12组尚未闭合，当前10组本身符合原生报告要求。

## A 实例接续

实例 `2112186448839770114`，SSH11909。新协调器 PID46904，源码 `/root/reschedule_kimi021.py`，SHA256 `c2465d268b42d7edc08bceb2803bfa3c95588d84aa75e13aaa69d5340a289eca`；进度 `/root/private_data/guoxin-session021/coordinator001/progress.json`，日志 `/root/kimi-bounded-continuation021.log`。

| 顺序 | 当前身份或路径 | 工作与边界 |
| --- | --- | --- |
| 1 | PF006，PID46978，UUID `96fe8634f55c497b838e77b1be01f917` | 只补16K/c64两轮，每轮128请求；单轮9000秒、总19000秒、父进程19100秒，服务看门狗仍21600秒 |
| 2 | `coordinator001/combined-native-matrix.json` | 严格核对旧10+新2的唯一完整12组/608条；只允许实际配置中的task路径和运行UUID不同，已完整报告不重新采样 |
| 3 | `kimi-portable-replay002` | 从Git `031051314c2a79ee76a9fbd57d363ba11244719b` 的原固定ZIP执行根run、env、smoke、自检、全1000条OCRBench，再停止自身服务 |
| 4 | Robo 原等待进程PID43964 | 协调器确认新旧Kimi控制器和服务完成清理后恢复，跑完整RefSpatial200，仅精度通过后跑性能 |
| 5 | AuK 原等待进程PID45258 | 保持原样等待同一Robo PID；Robo清理后生成完整400条×3轮、计算SIM，再由本地PID19812计算全部CPU CER |

PF006输出 `/root/private_data/guoxin-session021/kimi-performance006`。18:32:44 UTC实际服务状态为 `ready`、健康检查200；实际新旧服务配置逐键比较，仅task路径与UUID不同。原数学SDPA、BF16、max-len32768、max-num-seqs4、关闭prefix-cache与chunked-prefill均保持一致。

合成性能文本仍排除保留token，输出屏蔽四个输入占位符，须在报告披露；真实OCR请求不屏蔽。原001–004失败原件继续保存。原生客户端输出长度存在usage/fallback边界，不能宣称已独立验证线上每个token或获得官方验收。

## 等待进程的安全接续

为避免旧Kimi结束后Robo抢先启动，新协调器只对尚未GPU开始的自有Robo等待进程43964做了SIGSTOP。`/proc`状态T是预期等待，不代表死机。没有停止AuK进程，AuK仍识别同一Robo PID为存活。两个原队列的单调时钟等待期限均未改变。

协调器保存了每个PID的完整argv和启动token；其 `finally` 只在所有相关端口关闭、旧Kimi进程结束、自己的当前子进程退出后，核对身份并发送SIGCONT。即使新Kimi评测不合格，只要清理安全，也会释放Robo继续其他项目。

如果协调器异常消失且 `robo_released=false`，先检查 `current_child`、所有相关runtime和端口、现有进程的真实身份。确认新旧Kimi均完成清理后，才能按原启动token和完整argv恢复同一Robo PID；禁止盲目恢复、另开重复Robo/AuK或覆盖原实验文件。

## B 实例与其他边界

B `2112237945233408001`、SSH11911上的MiMo repair018继续三容器并行，未受本轮A调度影响。所有89项仍在分母；正常AgentTimeout的真实reward0保留，不能作为基础设施错误重试。WSL最终审计PID139578和隧道健康文件继续有效。

AuK的完整400输入、100秒输出上限、三轮seed、CER/SIM门槛及本地CPU评分规则继续按 `docs/active-queues-session019.md` 执行。原Robo和AuK失败分数不能升级为通过。

本轮启动回执、实际配置比较、源文件SHA、原始ZIP检查和安全测试均在 `evidence/session021`。完整Kimi矩阵、固定入口复现、最终报告、光合与镜像仍需真实结果闭合；本轮不增加完整交付计数。
