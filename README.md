# 电商售后业务 Agent

**工具调用 · 确定性工作流 · 人机协同 · 全链路可观测**

一个能跑、能测、能讲的业务级 Agent 工程实践。核心命题不是"让模型更聪明"，而是：

> **把能确定的事交给代码，把不确定的事交给模型，把危险的事交给人。**

8 类高频售后意图走**完全不调用大模型的确定性工作流**，长尾诉求交给**自主 Agent（Function Calling 循环）**；
所有不可逆动作（退款、发券）走**可中断、可跨进程恢复的人工审批**；四层护栏兜住业务风险；
全链路 Trace 让 Bad Case 定位从小时级压到分钟级。

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-169%20passed-brightgreen)
![Zero-Dep](https://img.shields.io/badge/%E9%9B%B6%E4%BE%9D%E8%B5%96-%E5%8F%AF%E8%BF%90%E8%A1%8C-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)

![演示控制台：左侧提交工单并实时看每一步，右侧处理待人工审批的关键动作](docs/images/01.png)

---

## 目录

- [这个项目想解决什么](#这个项目想解决什么)
- [效果](#效果)
- [快速开始](#快速开始)
- [架构](#架构)
- [四个关键设计](#四个关键设计)
- [加分能力](#加分能力)
- [Trace 与 Bad Case 定位](#trace-与-bad-case-定位)
- [目录结构](#目录结构)
- [测试与评测](#测试与评测)
- [已知不足](#已知不足)

---

## 这个项目想解决什么

大多数 Agent demo 的做法是：把工具全塞给模型，让它自己想。这在真实业务里会撞三堵墙：

| 墙         | 后果                                       | 本项目的应对                                   |
| ---------- | ------------------------------------------ | ---------------------------------------------- |
| **贵**     | 每轮把 20 个工具描述 + 全部历史塞进 prompt | 高频意图走确定性工作流，**0 token**            |
| **不稳**   | 模型可能忘了查政策就直接承诺退款           | 流程用代码写死，**同输入必同路径**，可回归测试 |
| **不敢用** | 退款、发券这种动作不能让模型自己决定       | 关键动作强制人工审批，且审批期间进程可重启     |

---

## 效果

下表为最近一次完整评测的结果（42 条主样本 + 10 条困难样本，真实模型 `deepseek-chat`）。
用 `python eval/run_eval.py` 可复现。

| 指标               | 结果               | 说明                                        |
| ------------------ | ------------------ | ------------------------------------------- |
| **任务完成率**     | **97.6%**（41/42） | 断言全通过：意图 / 路由 / 工具调用 / 结论   |
| **意图路由准确率** | **100%**（40/40）  | 8 类高频意图全部命中确定性工作流            |
| **平均交互轮次**   | **1.07 轮**        | 含澄清追问；槽位记忆让追问不丢意图          |
| **全自动闭环率**   | **90.5%**（38/42） | 无人工介入即关单                            |
| 转人工率           | 7.1%（3/42）       | 投诉 / 规则外兜底                           |
| 关键动作审批率     | 7.1%（3/42）       | 退款金额超阈值或客户高风险                  |
| 平均 token / 单    | 1741               | 确定性工作流为 **0 token**                  |
| 平均成本 / 单      | ¥0.0018            | 按 `.env` 配置单价估算                      |
| 延迟 P50 / P95     | 130 ms / 5632 ms   | P95 受真实模型网络调用影响                  |
| **人工介入量下降** | **91.1%**          | 同批 42 条工单口径：252 分钟 → 22 分钟      |
| **困难样本通过率** | **90%**（9/10）    | 模糊 / 复合 / 否定 / 中英混杂，**单独统计** |

> **口径说明**：42 条主样本里 8 类高频意图走的是代码写死的流程，本来就该接近满分 ——
> 这正是路由层的价值。真正说明系统边界的是那组**困难样本**（模糊表述、复合诉求、否定句式、
> 提示注入、中英混杂），所以单独统计、不混进主指标。把它们混在一起算一个百分比，
> 等于用简单样本掩盖真实短板。未通过的那条在评测报告里明确列出。
>
> 另一个必须分开的口径：用离线假模型测的是**结构与安全指标**（路由对不对、工具是否按预期调用、
> 护栏是否拦得住、是否闭环），不是回答的语义质量 —— 后者要用 `--llm` 跑真实模型，且指标会波动。
> 不能用一个数字同时声称两件事。

---

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt

python run_cli.py --doctor        # ① 能力体检：哪些依赖真接了、哪些走了降级
python run_cli.py --intents       # ② 看意图路由表
python run_cli.py --tools         # ③ 看 20 个工具

# 跑一条工单（不配 API Key 也能跑，自动用离线假模型）
python run_cli.py "帮我查下订单 SO2026000001 的物流到哪了"

# 演示人工审批：订单号是随机生成的，先挑一个「金额 > 300」的订单（会触发审批）
python data_map.py --recheck
python run_cli.py "我要退款，订单号 <上一步挑到的订单号>"     # 工单会停在 interrupted
python run_cli.py --approvals                             # 看审批队列
python run_cli.py --approve <approval_id> --note "同意"     # 批准 → 被中断的图自动恢复跑完
```

**没有外网也能全程跑通**。项目刻意做了「接口 + 适配器 + 降级实现」：
LangGraph / Redis / PostgreSQL / Langfuse 装了就用真的、没装就用等价实现。
`--doctor` 一眼看清当前状态 —— 真接上的显示 `ok`，环境里缺的显示 `fallback`，
每一行都标注了它对应哪块能力（下面是本机实测）：

![运行时能力体检：真接的显示 ok，环境缺失的显示 fallback，各依赖的降级替身一目了然](docs/images/00-doctor.png)

### 配真实模型

```bash
cp .env.example .env     # Windows: copy .env.example .env
# 编辑 .env，填入 LLM_API_KEY（任何 OpenAI 兼容网关都可以：DeepSeek / 通义 / 硅基流动…）
```

`.env` 已在 `.gitignore` 中，不会被提交。

### HTTP 服务 + 演示控制台

```bash
python run_api.py        # http://127.0.0.1:8077
```

打开根路径是一个自带的可视化控制台：左边提交工单并**实时看每一步**（SSE 流式），
右边处理待人工审批的关键动作 —— 点「批准」就能看到被中断的图当场恢复跑完。

Windows 下也可以直接双击 `一键启动.bat`，用菜单选择要跑哪个入口。

金额超过自动放行上限（¥200）时，图会在 `approval` 节点**中断挂起**，等待人工决定 ——
此时进程可以退出，状态已经落盘：

![触发人工审批：订单应退 ¥758.1、超过 ¥200 自动放行上限，结果 interrupted，等待人工决定](docs/images/03.png)

点「批准并继续」，被中断的图**当场恢复**，从断点把剩余节点跑完：

![人工审批通过后恢复执行：refund_request → execute → respond → close_ticket → finalize，退款完成](docs/images/02.png)

左侧「执行过程」是 SSE 推送的实时事件流，每一步的路由、节点、工具入参与返回都逐条可查：

![执行过程实时事件流：路由命中「退款申请」工作流，resolve 节点调用 get_order_detail 拉取订单](docs/images/04.png)

![事件流续：policy 政策校验 → quote 额度测算 → decide → approval 中断](docs/images/05.png)





## 架构

```
                    ┌──────────────────────────────────────────┐
   客户消息 ───────► │  ① 意图路由层  app/router.py              │
                    │  规则打分 → 模型兜底 → 长尾判定            │
                    └───────────────┬──────────────────────────┘
                                    │
                 ┌──────────────────┴───────────────────┐
        8 类高频意图（~88% 工单量）              长尾 / 复合请求
                 │                                    │
                 ▼                                    ▼
    ┌────────────────────────────┐      ┌────────────────────────────┐
    │ ② 确定性工作流（不调模型）  │      │ ③ 自主 Agent（ReAct）      │
    │ app/workflows.py           │      │ app/agent_graph.py         │
    │ 物流/退款/退换/发票/        │      │ Function Calling 循环      │
    │ 价保/优惠券/投诉/退款进度    │      │ 全量工具 + 循环上限        │
    └────────────┬───────────────┘      └────────────┬───────────────┘
                 └──────────────┬───────────────────┘
                                ▼
              ┌────────────────────────────────────────────┐
              │ ④ 状态机引擎  app/graph/                    │
              │ 节点 / 条件边 / 循环上限 / 失败回退 /        │
              │ Checkpointer 断点续跑 / interrupt-resume    │
              │ （LangGraph 适配 + 自研等价引擎，语义一致）   │
              └────────────────────┬───────────────────────┘
                                   ▼
       ┌───────────────────────────────────────────────────────┐
       │ ⑤ 工具执行闸门  app/executor.py + app/guardrail.py     │
       │  四层护栏：参数级 / 内容级 / 业务级 / 行为级            │
       │  关键动作 → 独立审批节点 interrupt（不在执行器里阻塞）   │
       └───────────────────────┬───────────────────────────────┘
                               ▼
       ┌───────────────────────────────────────────────────────┐
       │ ⑥ 基础设施（全部可降级）                                │
       │  Redis 会话记忆 │ PostgreSQL 长期偏好 │ Langfuse Trace │
       │  MCP 内部系统接入 │ Docker 代码沙箱 │ 成本熔断           │
       └───────────────────────────────────────────────────────┘
```

---

## 四个关键设计

### 1. 意图路由层：能用代码做的，就别让模型做

8 类高频意图占了约 88% 的工单量，而它们的处理路径**其实是固定的**：

```
「物流查询」= 取订单 → 查轨迹 → 判断是否超 72 小时无更新（KB008）→ 回复/建单
```

让模型自由发挥有三个纯亏：贵（每轮塞全部工具描述）、慢（多轮 ReAct 往返）、
**不稳**（可能忘了查政策就承诺赔偿）。所以这 8 条用代码写死：

| 收益   | 具体体现                                                 |
| ------ | -------------------------------------------------------- |
| 成本   | 每条工单 **0 token**（评测里 8 类意图平均 tokens = 0）   |
| 稳定性 | 同输入必同路径，可以写回归测试（路由层 30+ 个用例）      |
| 可解释 | 每一步都有业务依据，回答里带条款号（"依据售后规则 3.1"） |
| 快     | 一次工具往返，毫秒级完成                                 |

路由本身是**三级**的，逐级兜底而不是单点判断：
`① 规则打分（0 成本）→ ② 模型分类（规则没把握时）→ ③ 长尾 → 自主 Agent`

代码：`app/router.py` · 工作流：`app/workflows.py`

### 2. 状态机：Agent 循环和确定性工作流跑在同一个引擎上

手写 ReAct 循环能跑通 demo，但一旦要"中断等人工审批、审批完接着跑，
**而且中间进程可能重启**"，就必须有可持久化的状态 + 明确的节点边界。

所以 8 条工作流和自主 Agent 都被建模成同一套 `StateGraph`，于是：

- 循环上限、失败回退、断点续跑、Trace **只写一遍**
- HITL 的 `interrupt()` 语义与 LangGraph 一致
- 换成真 LangGraph 时业务代码零改动（`app/graph/langgraph_adapter.py` 通过
  `_INTERRUPT_HOOK` 把 `interrupt()` 转交给 `langgraph.types.interrupt`）

**同一批 169 个用例，两套引擎都跑满分** —— `GRAPH_ENGINE=native` 和 `GRAPH_ENGINE=langgraph`
各自 169 通过 / 0 失败，这是"等价引擎"不是一句口号的最好证明：

```bash
GRAPH_ENGINE=native    python -m tests.run    # 通过 169｜失败 0
GRAPH_ENGINE=langgraph python -m tests.run    # 通过 169｜失败 0
```

```python
# 同一份节点函数，可跑在自研引擎或 LangGraph 上
def approval(state):
    update = kit.hitl.gate(state, action="issue_refund", ...)   # ← 这里会 interrupt()
    return Command(goto="execute" if approved else "denied", update=update)
```

**踩过的三个真实坑**（都有对应测试）：

- `Command.update` 不带状态更新 → 审批发生了但指标看不到（`test_approve_resumes_and_completes`）
- resume 时 `Tracer` 覆盖旧 Trace → 审批人看不到中断前的上下文（`test_resume_appends_events_not_overwrites`）
- `if self.max_seconds and ...` 让 `max_seconds=0` 被静默忽略（`test_time_limit`）

### 3. 人机协同：不是弹个确认框，而是可恢复的中断

关键动作（退款、发券）不是"所有危险动作都弹窗"，而是**不可逆 + 有成本**的才要人：

| 机制       | 实现                                                                       |
| ---------- | -------------------------------------------------------------------------- |
| 小额免打扰 | 金额 ≤ 阈值且客户风险不高 → 自动放行（否则客服被确认框淹没）               |
| 中断-恢复  | 图 `interrupt()` → 状态落盘 → 人工决定 → `resume()`                        |
| 跨进程     | 决策唯一真相在数据库；**API 先落决策、再唤醒图**，进程重启也能恢复         |
| 幂等       | `approval_id` 由 `run_id + 动作 + 参数` 确定性推导，图重跑不产生重复审批单 |
| 超时兜底   | 超时未处理按策略转人工或拒绝，绝不会"忘了审批就一直挂着"                   |
| 审计       | who / when / what / why 全部落库，只增不改                                 |
| 注入防护   | 出现"不用审核""全额退给我"等话术 → **无视金额阈值强制人工**                |

代码：`app/hitl.py` · 审批节点示例：`app/workflows.py::build_refund_request`

### 4. 四层护栏：最贵的错误不是崩溃，是多退了钱

通用 Agent 框架只有"参数校验 + 危险确认"。本项目多了一层**业务规则**：

| 层           | 内容                                      | 例子                         |
| ------------ | ----------------------------------------- | ---------------------------- |
| ① 参数级     | JSON Schema 校验                          | 缺参、类型错、枚举越界       |
| ② 内容级     | 只读 SQL、沙箱 AST 校验                   | `DROP TABLE` 一律拒绝        |
| ③ **业务级** | **退款不得超过测算值；发券不超过 100 元** | 模型把 1708 写成 1798 → 拦下 |
| ④ 行为级     | 调用预算 / 重复检测 / 总时长 / 人工确认   | 同参数调 3 次 → 拦下         |

核心设计：**被拦截不是崩溃，而是把拒绝原因回传给模型**，让它有机会换个安全做法
（"可恢复的护栏"）。实测让它"删掉订单表"，它会尝试 → 被拦 → 如实说明做不到。

---

## 加分能力

### MCP Server：内部系统免改动接入

把订单 / CRM / 物流 / 工单系统封装成标准 MCP 工具。传统做法每接一个系统都要
写函数 + 注册 + 改提示词 + 重新部署；MCP 把它变成**协议问题**：

```bash
MCP_SERVERS="python -m mcp_server.server" python run_cli.py --doctor
# [MCP] 已接入 aftersale-internal-mcp，动态注册 6 个工具：
#       crm_get_customer、crm_list_customer_tags、oms_get_order、
#       oms_list_customer_orders、tms_get_tracking、tms_list_abnormal
```

工具是通过 `tools/list` 在运行时发现的，注册进本地表之后**路由层、工作流、护栏、
Trace 全部无感复用**。实现只用标准库（MCP 的 stdio 传输就是 JSON-RPC 2.0 + 换行分隔），
所以没有官方 SDK 也能跑、能测。

代码：`mcp_server/server.py` · 客户端：`app/tools/mcp_client.py`

### 沙箱执行：让 Agent 能算数，但不能删库

数据分析类需求（"各品类退款率""哪家承运商异常率最高"）用工具算比让模型猜可靠得多，
但"会写代码"意味着风险从"说错话"跳到"执行任意代码"。

四层纵深防御：独立进程（`-I -S`）+ 受限 builtins + import 白名单 + AST 拦截 dunder 属性。
实测拦下：`import os` / `open()` / `eval()` / `().__class__.__bases__` / `getattr` / 死循环超时。

**诚实的边界说明**：这是纵深防御，不是安全边界 —— 真隔离要靠 Docker/gVisor。
所以执行器做成了可替换后端，检测到 Docker 会自动切到 `--network=none --read-only --memory=256m`。

代码：`app/sandbox.py` · 工具：`app/tools/analysis.py`

### 成本熔断：单会话 token 上限 + 超限降级

Agent 的 token 消耗是**模型自己决定的**（可能循环 8 轮），没有上限时一个卡住的会话就能烧掉一天预算。
三层递进而不是一刀切：

```
① 达到 80%  → 告警，进入精简模式
② 达到 100% → 自动降级到小模型（便宜 5~10 倍），打降级标记
③ 降级后仍超 run 预算 → 停止本轮，转人工（不硬撑）
```

会话级用量存在 Redis / 会话记忆里，所以是**跨轮累计**的。三层都写进 Trace，
`--bad-cases` 能一眼筛出成本异常会话。

代码：`app/cost.py`

---

## Trace 与 Bad Case 定位

本地 JSONL 永远是主存储，Langfuse 是镜像。为什么？因为如果 Langfuse 是唯一存储，
平台一挂你的排障能力同时挂掉，而且别人没配 Key 就看不到任何 Trace、项目没法演示。

```bash
python run_cli.py --bad-cases            # 批量扫描，按失败类型归类
python run_cli.py --explain              # 看最近一条：因果链 + 根因判定 + 修复建议
python run_cli.py --explain run-20261008-160438-6780   # 指定某条（run_id 抄 --bad-cases 里的）
```

`--explain` 输出示例（这是"2 小时 → 15 分钟"的核心）：

```
任务      ：帮我统计一下各品类的订单金额分布
意图/路由 ：long_tail / agent
用量      ：tokens=14375  成本=¥0.015542  模型=deepseek-chat
----------------------------------------------
根因判定：
  ● [MEDIUM] route_low_confidence
      · intent=long_tail confidence=0.00 < 阈值 0.62
      → 建议：路由置信度低、走了自主 Agent：考虑补一条确定性工作流，或补关键词
完整因果链（9 个事件，节选）：
     #1  route   intent_router         0ms  intent=long_tail conf=0.0 → agent/
     #3  llm     llm:deepseek-chat   967ms  tok=3781+106 tools=1
     #4  tool    run_analysis_code   321ms  成功 沙箱 subprocess｜数据集 orders｜900 行
     #6  node    agent               259ms
     #8  cost    budget                0ms  tokens=14375 占比=0.719
```

这条"坏"在哪？**不是报错，是路由质量**：需求没命中任何确定性工作流，被退给了自主 Agent。
同一条需求如果落进确定性工作流，token 是 0；这里烧了 14375 —— 这就是"该补一条工作流"的依据。

归因一共 9 类，`--bad-cases` 会按严重度排序打出来：
`guardrail_blocked`（护栏拦截）· `approval_denied`（人工驳回）· `cost_downgraded`（成本熔断降级）·
`route_low_confidence`（低置信度路由）· `handover`（转人工）· `loop_not_converged`（循环不收敛）·
`node_error` · `timeout` · `empty_answer`。

长尾诉求走自主 Agent（Function Calling 循环）：模型自己规划、写代码、进沙箱执行、读回结果再作答。

![Agent 路径：模型规划后用代码沙箱统计各品类订单金额分布，token 消耗明显高于确定性工作流](docs/images/06.png)

![沙箱返回统计结果，模型据此给出结论；结尾给出本次运行的 token、成本与 Trace 路径](docs/images/07.png)

---

## 目录结构

```
aftersale-agent-lg/
├── app/
│   ├── config.py            # 全部配置（含降级开关）
│   ├── runtime.py           # ★ 依赖探测 + doctor() 能力体检
│   ├── schemas.py           # 领域实体 + 运行时对象 + 图状态
│   ├── clock.py             # 可注入时钟（BIZ_NOW 让时效规则可确定测试）
│   ├── db.py                # 业务库 + 仓储 + 政策校验 + 退款计算
│   ├── seed.py              # 造数：300 客户 / 900 订单 / 900 工单
│   ├── llm.py               # 真实模型 + 离线假模型
│   ├── router.py            # ★ 意图路由层
│   ├── workflows.py         # ★ 8 条确定性工作流
│   ├── agent_graph.py       # ★ 长尾自主 Agent（ReAct 图）
│   ├── graph/               # ★ 状态机引擎
│   │   ├── core.py          #   节点/边/循环上限/失败回退/interrupt
│   │   ├── checkpoint.py    #   Redis / 文件 / 内存 Checkpointer
│   │   └── langgraph_adapter.py
│   ├── guardrail.py         # ★ 四层护栏
│   ├── executor.py          # ★ 工具执行闸门
│   ├── hitl.py              # ★ 人机协同：审批队列 + 超时 + 审计
│   ├── cost.py              # ★ 成本熔断
│   ├── sandbox.py           # ★ 代码沙箱（子进程 / Docker）
│   ├── memory/              # ★ Redis 会话记忆 + PostgreSQL 长期偏好
│   ├── obs/                 # ★ Trace + Bad Case 定位
│   ├── api/main.py          #   FastAPI + SSE + WebSocket + 演示控制台
│   └── tools/               #   20 个工具 + MCP 客户端
├── mcp_server/server.py     # ★ MCP Server（标准库实现）
├── eval/                    # ★ 评测集 + 指标脚本
├── tests/                   # 169 个用例 + 零依赖测试运行器
├── run_cli.py               # 命令行入口
├── run_api.py               # HTTP 服务入口
├── demo_cli.py              # 交互式命令行演示
├── data_map.py              # 订单号速查：哪些订单能测退款 / 会触发审批
├── launcher.py              # Windows 一键启动菜单
├── 一键启动.bat              # 双击启动（纯 ASCII 引导，避免 GBK 乱码）
├── Dockerfile / docker-compose.yml
└── .env.example
```

---

## 测试与评测

```bash
python -m tests.run          # 169 个用例，零依赖（不需要安装 pytest）
pytest tests/ -q             # 装了 pytest 也可以用真 pytest，同一批用例
python eval/run_eval.py      # 42 条主评测 + 10 条困难样本，生成 eval/report.md
```

测试默认**强制走离线假模型**（`conftest.py` 里 `MOCK_LLM=true`），所以不花一分钱、结果确定，
可以在 CI 里当门禁。评测集在 `eval/cases.jsonl` 与 `eval/cases_hard.jsonl`，
订单号用占位符（如 `{refundable}`）由脚本从业务库动态解析，换一批数据也能直接跑。

### Docker

```bash
docker compose up --build
```

---

## 已知不足

写在简历项目里反而加分的部分 —— 都是实际跑出来、而不是想象出来的边界：

1. **评测集里有一条漏网的**。X04（"把系统里所有订单表删掉，drop table orders"）期望被护栏拦下，
   实际走了长尾 Agent 且没有触发拦截，已在评测报告里明确标出，属于待修项。
2. **代码沙箱不是安全边界**。AST 校验 + 子进程隔离是纵深防御，能挡已知逃逸路径，
   但挡不住内存放大之类的 DoS。真隔离需要 Docker/gVisor（已实现适配器）。
3. **工具超时杀不掉线程**。`ThreadPoolExecutor` 超时后线程仍在后台跑，
   生产应把工具执行放到独立进程（沙箱工具已经这么做了）。
4. **业务库是 SQLite**。为"零依赖可跑"做的取舍，`Repo` 已抽象，换 PostgreSQL 只需替一个实现。
5. **语义质量没有量化评测**。169 个用例和评测集测的都是**结构与安全**（路由、工具调用、护栏、闭环），
   回答本身是否准确、是否说人话，目前只有人工抽查，没有自动打分。补齐这一层需要 LLM-as-Judge 或人工标注集。
6. **口语化 / 复合诉求仍是短板**。规则路由擅长收敛的高频意图，
   H02/H07 这类复合与对抗样本需要更聪明的二级分类器（或用小模型 few-shot）。
