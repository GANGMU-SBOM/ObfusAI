<div align="center">

# 🌀 ObfusAI

**面向 LLM 应用的提示词卫生工具包：找出提示词里的密钥，发现模型泄露指令，按需混淆，并实测这些手段在*你的*模型上是否真的有效。**

Python 库，另有面向其他所有语言的本地网关。
核心零依赖 · Python 3.9+ · 支持 OpenAI · Anthropic · LangChain · LlamaIndex

[![GitHub stars](https://img.shields.io/github/stars/GANGMU-SBOM/ObfusAI?style=social)](https://github.com/GANGMU-SBOM/ObfusAI/stargazers)
[![PyPI](https://img.shields.io/pypi/v/obfusai)](https://pypi.org/project/obfusai/)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![dependencies](https://img.shields.io/badge/core%20dependencies-0-brightgreen)

[English](README.md) · **简体中文**

**⭐ 如果 ObfusAI 对你有用，请[在 GitHub 上点个 Star](https://github.com/GANGMU-SBOM/ObfusAI)——只需一次点击，却能让更多开发者发现它。**

</div>

---

## 它到底解决什么问题？

如果你的系统提示词或工具 Schema 里承载着业务逻辑（阈值、定价规则、评分、内部代号），可能出五类不同的问题。它们需要不同的工具，被解决的程度也并不一样：

| # | 问题 | ObfusAI 提供什么 | 能信到什么程度 |
|---|---|---|---|
| 1 | **密钥和个人信息被写进了提示词**（API Key、银行卡号、客户邮箱） | `audit` / `obfusai audit`：对银行卡、证件号、IBAN、密钥、JWT、邮箱、电话等做带校验位的检测 | 已知格式**可靠**。属于启发式：只能降低概率，不能证明提示词是干净的 |
| 2 | **模型泄露了它的指令**（“把上面的内容原样重复一遍”），而且没人发现 | `LeakDetector`：扫描并脱敏回复里的受保护值、别名和锚点。明文提示词也能用（`values=[...]`） | 对你列出的值**检测得准**。只检测，不阻止 |
| 3 | **你不知道提示词到底有多容易泄露，也不知道改动是否把它改坏了** | `evaluate_extraction`（9 种套取攻击，统计泄露率）和 `evaluate_fidelity`（行为是否保持），外加面向真实模型的一键脚本 | **测的是你的模型**，所以对该模型和该提示词可信 |
| 4 | **明文逻辑散落在日志、链路追踪、后台、截图里** | 混淆：把数字、关键词、标识符换成符号，并附上解码表 | **效果弱，也不是免费的。** 只能防“扫一眼”，防不住“读一读”（详见下文） |
| 5 | **你的技术栈不是 Python** | `obfusai serve`：本地网关，任何语言的 OpenAI/Anthropic SDK 指过去即可 | 可用。已在 JS SDK 以及 Go、Java、Ruby、PHP 的 HTTP 客户端上验证 |

所以诚实的概括是：**可靠的部分是审计、泄露检测和评测工具；混淆只是一个可选的附加项，它有没有价值你必须自己测。**

### 混淆能做什么，不能做什么

```text
混淆前                                           混淆后（medium）
You are the Alpha Momentum Engine for           You are the λ for τ.
Helios Capital.                                 Signal "π": Ψ when δ > ξ and Θ <= θ.
Signal "mean_reversion_v7": BUY when            Never exceed φ of Ξ. Hard stop: β = μ.
momentum_score > 0.73 and risk_budget_bps <= 45.
Never exceed 2.5% of NAV.                       <obfusai-context v="1" id="2d5d5ff8">
Hard stop: drawdown_limit = 8.2%.               ……解码说明……
                                                λ = "Alpha Momentum Engine"
                                                δ = "momentum_score"   ξ = "0.73"   ……
                                                </obfusai-context>
```

- 符号替换了正文里的值，但**解码表就在同一条消息里**。`low` / `medium` 档里，解码表以明文列出真实值，所以搜索 `momentum_score` 仍然能找到（在表里）。`high` 档把定义逆序并混入诱饵，能挡住 `grep` 和随意浏览，但挡不住真人或模型。
- 会重复上下文的模型会连解码表一起重复。仓库里的单元测试（`tests/test_redteam.py`）里，回显型模型无论用不用 ObfusAI，受保护的值都 100% 泄露。
- 它不是免费的：短提示词会因为解码表膨胀约 3～4 倍，模型按解码后的指令执行也可能不如按明文执行。
- 适用场景：提示词必须承载逻辑，*并且*你在意的风险是随意暴露（有人扫一眼链路追踪、被截了一半的截图）。不要把它当作保密措施。

### 它不解决什么

- **抵御有决心的攻击者的保密性。** 这是混淆，不是加密，更不是安全边界。
- **凭证。** 永远不要去混淆它们，直接删掉（`audit` 能找出来）。
- 如果你需要真正的保密，请改架构：把逻辑放在服务端，让模型调用只返回结果、不返回规则的工具；永远别把机密写进提示词；做限流并过滤输出。ObfusAI 是对这些措施的补充，不是替代。

> **依赖这一切之前，先实测：** `python examples/validate_real_model.py --yes`（需要你自己的 API Key，见[上线前先实测](#上线前先实测)）。这个仓库里的东西目前还没有用真实模型实测过。

## 功能一览

| | |
|---|---|
| 🔎 **提示词审计** | `audit()` / `obfusai audit` 找出提示词里的密钥、银行卡、证件号、IBAN、电话、邮箱、IP，带校验位验证，预览已脱敏 |
| 🛡 **泄露防护** | `LeakDetector` 扫描并脱敏模型输出里的受保护值、别名和锚点；用不用混淆都能用 |
| 🧪 **红队 + 保真度评测** | `evaluate_extraction` 与 `evaluate_fidelity` 在你的模型上测泄露率和行为，对比明文与混淆后的效果 |
| 🌍 **任何语言** | `obfusai serve` 网关：把任意语言（JS/TS、Go、Java、C#、Rust、Ruby、PHP……）的 OpenAI/Anthropic SDK 指过去即可；JSON 与 SSE 回复都会还原 |
| 🔌 **Python 集成** | OpenAI（Chat Completions 与 Responses API）和 Anthropic 包装器（同步、异步、流式），LangChain 与 LlamaIndex 适配器 |
| 🔣 **混淆（可选）** | 数字、标识符、引号字符串、缩写、邮箱、URL、IP、日期、你的关键词/正则 -> 希腊字母别名；三档强度；按字段的策略；用于提示词缓存的稳定别名；工具调用参数和流式输出会被还原 |
| ⚓ **上下文锚定** | 自带 `<obfusai-context>` 说明块（中文或英文，完整或精简），告诉模型如何解码 |
| 🪶 **零依赖核心** | 纯标准库、严格类型标注，遇到空串或怪异输入也不会崩溃 |

## 安装

```bash
pip install obfusai                    # 核心，无任何依赖
pip install "obfusai[langchain]"       # 含 LangChain 适配器
pip install "obfusai[llamaindex]"      # 含 LlamaIndex 适配器
# 安装最新开发版：
pip install git+https://github.com/GANGMU-SBOM/ObfusAI.git
```

## 快速开始

先用最可靠的部分，按这个顺序来。

**1. 审计提示词里有没有密钥和个人信息**（适合放进 CI：发现问题时退出码为 1）

```bash
obfusai audit prompt.txt
```
```python
from obfusai import audit

report = audit(system_prompt)
if not report.clean:
    print(report.counts)        # {"secrets": 1, "emails": 2}，预览已脱敏
```

**2. 防止模型回复里出现你绝不希望用户看到的值**

```python
from obfusai import LeakDetector

guard = LeakDetector(values=["mean_reversion_v7", "0.73", "Helios Capital"])   # 不需要混淆
if guard.scan(model_reply).leaked:
    model_reply = guard.redact(model_reply)
```

**3. 在你的模型上测一测提示词到底有多容易泄露**（需要你的 API Key）

```bash
python examples/validate_real_model.py --yes
```

**4. 混淆（可选）**，只有实测表明有帮助时再用：

```python
from obfusai import ObfusAIComponent

engine = ObfusAIComponent("medium", seed="my-secret", keywords=["Helios Capital"],
                          anchor_language="zh")          # 解码说明用中文

result = engine.obfuscate(system_prompt)
result.prompt                       # 混淆文本 + 锚点，直接作为 system 消息发送
result.restore("η 是主要驱动因素")    # 还原回复里出现的别名
result.stats.growth                 # 输出体积 ÷ 原文体积
```

### 保护完整的对话请求

```python
protected = engine.obfuscate_payload({
    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
    "tools": TOOLS,
})
client.chat.completions.create(model="gpt-4o", **protected.payload)
```

### 或者直接包装 SDK 客户端

```python
from openai import OpenAI
from obfusai.adapters.openai import ObfusAIOpenAI

client = ObfusAIOpenAI(OpenAI(), ObfusAIComponent("high", seed="my-secret"))
reply = client.chat.completions.create(model="gpt-4o", messages=[...], tools=[...])   # 流式同样支持
reply = client.responses.create(model="gpt-4o", instructions=SYSTEM, input="hi", tools=[...])  # Responses API
print(reply.output_text)                                                              # 已还原
parsed = client.responses.parse(model="gpt-4o", instructions=SYSTEM, input="hi", text_format=Signal)   # 结构化输出

with client.responses.stream(model="gpt-4o", instructions=SYSTEM, input="hi") as stream:   # 流式辅助
    for event in stream:                     # delta / snapshot / done 事件均已还原
        ...
    final = stream.get_final_response()      # 已还原
```

```python
from anthropic import Anthropic
from obfusai.adapters.anthropic import ObfusAIAnthropic

client = ObfusAIAnthropic(Anthropic(), ObfusAIComponent("medium", seed="my-secret"))
reply = client.messages.create(model="claude-sonnet-5-5", system=SYSTEM, messages=[...], max_tokens=512)

with client.messages.stream(model="claude-sonnet-5-5", system=SYSTEM, messages=[...], max_tokens=512) as stream:
    for text in stream.text_stream:        # 已还原；别名被切在两个事件之间也没问题
        print(text, end="")
    # 也可以直接迭代 SDK 事件（原始事件、`text`、`content_block_stop`），同样已还原；每个流只用一种消费方式
```

> 包装器保护 `chat.completions.create`、`responses.create` / `parse` / `stream`、`messages.create` 与 `messages.stream`。其他接口（批处理、旧版 completions 等）**不受保护**：未受保护的 `messages.*` 方法会直接报错而不是悄悄发出明文，其他客户端属性则原样转发。字典形式的结构化输出 Schema（`text`、`response_format`）会像工具 Schema 一样被混淆；以 pydantic 类给出的 Schema（`text_format`）由 SDK 转换后原样发送，`output_parsed` 由 SDK 根据模型的原始文本解析。

### LangChain

```python
from obfusai.adapters.langchain import ObfusAIDocumentTransformer, ObfusAIProtectedModel, ObfusAIRunnable

chain = prompt | ObfusAIProtectedModel(chat_model, ObfusAIComponent("high"))   # 混淆 → 调用 → 还原（含流式）
chain = prompt | ObfusAIRunnable(ObfusAIComponent("high")) | chat_model         # 仅混淆
docs  = ObfusAIDocumentTransformer(ObfusAIComponent()).transform_documents(docs)  # 保护 RAG 里的内部手册
```

### 任何语言：网关

把引擎移植到每一种语言，各个版本迟早会各走各路（Go/Rust 的正则还不支持检测器依赖的环视），所以只保留一份实现，并在前面放一个网关：

```bash
pip install obfusai
obfusai serve --port 8787 --level medium --seed "$OBFUSAI_SEED"     # 然后只改 SDK 的 base URL
```

| 语言 / SDK | OpenAI | Anthropic |
|---|---|---|
| JavaScript / TypeScript | `new OpenAI({ baseURL: "http://127.0.0.1:8787/v1" })` | `new Anthropic({ baseURL: "http://127.0.0.1:8787" })` |
| Go | `option.WithBaseURL("http://127.0.0.1:8787/v1/")` | 同一选项，地址 `http://127.0.0.1:8787` |
| Java / Kotlin | `.baseUrl("http://127.0.0.1:8787/v1")` | `.baseUrl("http://127.0.0.1:8787")` |
| C# / .NET | `new OpenAIClientOptions { Endpoint = new Uri("http://127.0.0.1:8787/v1") }` | 基础地址 `http://127.0.0.1:8787` |
| Rust | `OpenAIConfig::new().with_api_base("http://127.0.0.1:8787/v1")` | 基础地址 `http://127.0.0.1:8787` |
| Ruby · PHP · Swift · curl …… | 任何可配置 base URL 的客户端 | |

网关处理 `POST /v1/chat/completions`、`/v1/responses` 和 `/v1/messages`（含 `count_tokens`）：请求被混淆，回复被还原（工具调用参数和 SSE 流式都包括）。转发的是你自己的 `Authorization` / `x-api-key` 头，网关不保存任何密钥。其他接口会被**不加保护地**原样转发（会打印警告；`--strict` 会直接拒绝这类 POST）。非法 JSON 会被拒绝，绝不转发。

**已验证的范围：** 官方 OpenAI 与 Anthropic 的 JavaScript SDK，Go、Java、Ruby、PHP 的纯 HTTP 客户端（见 `examples/gateway/`，用 `run_demo_stack.py` 无需 API Key 即可对着假的供应商运行），以及测试套件里的 Python SDK。上表中 C#、Rust 等其他 SDK 的选项名来自它们的文档，**没有在这里运行过**。网关是面向 localhost 的标准库 sidecar，不是面向公网的代理：没有认证、每个请求一条连接、吞吐一般。

### LlamaIndex

```python
from obfusai.adapters.llamaindex import ObfusAILLM, ObfusAINodePostprocessor

llm = ObfusAILLM(OpenAI(model="gpt-4o"), ObfusAIComponent("high", seed="my-secret"), system_prompt=SYSTEM)
engine = index.as_query_engine(
    llm=llm,                                              # 系统提示词被混淆，回复自动还原（流式/异步同样支持）
    node_postprocessors=[ObfusAINodePostprocessor()],     # 检索到的片段在检索*之后*再混淆
)
```

建议在查询阶段用后处理器混淆，这样检索仍然基于可读文本；`ObfusAITransform`（入库阶段）会把混淆后的文本存进索引，向量检索效果会变差。请把 `system_prompt` 设在包装器上，而不是被包装的 LLM 上。`ObfusAILLM` 本身就是 `FunctionCallingLLM`：`chat_with_tools`、`predict_and_call`、Agent 以及各流式变体都能用，字典形式的工具 Schema（OpenAI、Anthropic、Mistral 类提供商）里的 `description` / `title` 文本会与系统提示词共用同一张别名表被混淆（名称、参数、枚举保持不变）。节点元数据只有在 `obfuscate_metadata=True` 时才会被混淆。

## 三档强度

| 档位 | 混淆内容 | 取舍 |
|---|---|---|
| `low` | 数字、百分比、货币、日期与时间（一个日期只生成一个别名）、snake/camel/Pascal/点号标识符、自定义关键词与规则 | 锚点最小，可靠性最高 |
| `medium`（默认） | + 引号内字符串、ACRONYM 缩写、Proper Noun 专有名词短语 | 均衡 |
| `high` | + 领域长词、`⟦α⟧` 包裹式别名、**逆序**锚点定义、**诱饵**别名、乱序表 | 锚点最大、最难读，模型解码余量最小——**务必在你的模型上实测** |

只有 `high` 档能让值躲过纯文本搜索：`low` 和 `medium` 档的锚点里是逐字列出的。

始终保持原样：模板占位符（`{var}`、`{{var}}`、`${var}`、`%(x)s`）、聊天特殊标记、列表序号，以及 `NEVER` / `MUST` 这类指令性词汇。短提示词要承担固定的锚点开销（上面的例子体积约 ×4）；真实的多页提示词里这个比例会迅速下降，`anchor_style="compact"` 还可进一步压缩。

## 精细控制要隐藏什么

```python
ObfusAIComponent(
    "medium",
    keywords=["Helios"],                       # 一律隐藏（不区分大小写）
    protected=["JSON"],                        # 一律不隐藏
    rules=[r"ACME-\d{4}", r"project (\w+)"],   # 自定义正则（带捕获组时只替换捕获组）
    extra_detectors=["emails", "urls", "ips"], # 可选检测器，见 obfusai.DETECTORS
    alias_mode="stable",                       # 相同原值 → 在所有提示词里得到相同别名
    anchor_language="zh", anchor_style="compact",
)
```

### 个人信息与凭证

可选的检测器，带校验位验证（过不了 Luhn 校验的 16 位数字*不会*被当成银行卡）：

| 检测器 | 匹配内容 |
|---|---|
| `phones` | 带 `+`、括号、连字符或点号的国际/国内号码 |
| `cn_phones` | 中国大陆手机号（`1[3-9]xxxxxxxxx`，可带 `+86`） |
| `cards` | 银行卡（Luhn）· `ssn` 美国社保号 · `cn_ids` 18 位居民身份证（校验位+出生日期）· `ibans`（mod-97） |
| `uuids` · `secrets` | UUID · AWS/GitHub/Slack/Google/`sk-…` 密钥、JWT、私钥块 |

```python
ObfusAIComponent("medium", extra_detectors=["phones", "cards", "cn_ids", "uuids"])
```

> 锚点里仍然带着每一个被隐藏的值，所以**凭证应当直接从提示词里删除，而不是混淆。** 用审计功能把它们找出来：

```python
from obfusai import audit

report = audit(system_prompt)          # 默认检测邮箱、IP、密钥与个人信息
if not report.clean:
    print(report.counts)               # {"secrets": 1, "emails": 2}
    for f in report.findings:          # 检测器、起止位置和*脱敏后*的预览（绝不输出原值）
        print(f.detector, f.start, f.end, f.preview)
```

命令行 `obfusai audit prompt.txt` 效果相同（发现问题时退出码为 1）。检测基于启发式规则：它能降低把敏感数据发出去的概率，但不能证明提示词是干净的。

### 按字段的策略

请求的不同部分值得区别对待：系统提示词里的专有数字要隐藏，而如果你的模型在工具描述可读时选工具更准，就让工具描述保持原样。

```python
from obfusai import FieldPolicy

engine = ObfusAIComponent(
    "high",
    policies={
        "system": FieldPolicy(extra_detectors=["emails"], keywords=["Helios"]),  # 仅对该字段追加
        "tools":  FieldPolicy(skip=True),                                         # 完全不动
        "*":      FieldPolicy(detectors=["numbers"]),                             # 其他字段的兜底策略
    },
)
```

字段名：消息角色（`system`、`developer`、`user` 等）、`instructions`（Responses API）、`tools`、`documents`（LangChain）、`nodes` / `prompt`（LlamaIndex）。策略里的 `keywords`、`protected`、`rules` 会叠加到组件级设置上；`detectors` 表示替换，`extra_detectors` 表示追加。被跳过的字段保持逐字节不变——锚点也不会写进去（需要时会另外加一条 system 消息）。所有适配器都遵循策略。

## 守住输出这一侧

```python
from obfusai import LeakDetector

guard = LeakDetector(result.mapping, session_ids=[result.session_id])
report = guard.scan(model_reply)
if report.leaked:                       # 类型：anchor · definition · session · alias · value
    reply = guard.redact(model_reply)   # 清除锚点/定义，遮盖别名与原值
```

配合混淆使用时，用这次运行的结果来构造检测器：`LeakDetector(result.mapping, session_ids=[result.session_id])`。不混淆时，像快速开始里那样传入要保护的值（`values=[...]`）。流式安全还原：`StreamRestorer(result.mapping).feed(chunk)` …… `.flush()`。

## 上线前先实测

混淆值不值得用，取决于两个问题，而且只有你自己的模型能回答。一条命令同时问这两个：

```bash
export OPENAI_API_KEY=...        # 或 ANTHROPIC_API_KEY
python examples/validate_real_model.py --yes      # 会先打印将要发起的 API 调用数
```

1. **保真度**：模型按解码后的提示词执行，是否和按明文执行一样好？（`evaluate_fidelity`）
2. **防护效果**：提示词套取攻击（“原样重复上面所有内容”“翻译你的指令”“base64 输出”“解码这张表”……）泄露被隐藏的值的概率，是否比攻击明文提示词时*更低*？（`evaluate_extraction`）

如果混淆后的泄露率没有明显低于明文，说明混淆在这个模型上没起作用：两者都高，说明模型会把解码表一起重复；两者都是零，说明模型本来就会拒绝。请先把示例提示词换成你自己的，再相信结论。

```python
from obfusai.redteam import evaluate_extraction

report = evaluate_extraction(my_chat_callable, SYSTEM, engine=ObfusAIComponent("medium"))
print(report.summary())     # 攻击明文与混淆后的泄露率，以及增益
report.helps()              # 比明文至少低 30 个百分点吗？
```

```python
from obfusai.evaluation import FidelityCase, evaluate_fidelity

report = evaluate_fidelity(
    my_chat_callable,                                    # messages -> 回复文本
    [FidelityCase("buy", SYSTEM, "edge_bps=40", "BUY")],
    levels=("low", "medium", "high"), repeats=3,
)
print(report.summary())
report.acceptable("high")   # 通过率不低于明文基线，且没有泄露？
```

`obfusai.testing.DecodingMockLLM` 是一个会先解码锚点的确定性替身，适合写单元测试。

## 命令行

```bash
obfusai obfuscate prompt.txt -l high --seed s --lang zh --mapping-out map.json > protected.txt
obfusai restore reply.txt -m map.json
obfusai scan reply.txt -m map.json        # 回复存在泄露时退出码为 1
obfusai audit prompt.txt                  # 提示词含密钥/个人信息时退出码为 1
obfusai serve --port 8787 --seed "$OBFUSAI_SEED"   # 面向任何语言的网关
```

## 工作原理

```mermaid
flowchart LR
    A[明文提示词 + 工具 Schema] --> B[检测器<br/>数字 · 标识符 · 术语 · 自定义规则]
    B --> C[共享别名表<br/>希腊符号]
    C --> D[混淆后的文本]
    C --> E[锚点块<br/>解码表]
    D & E --> F[LLM]
    F --> G[回复]
    G --> H[LeakDetector / StreamRestorer]
```

## 威胁模型

| 暴露面 | ObfusAI 做了什么 |
|---|---|
| 日志、链路追踪、供应商后台、截图里的提示词副本 | 数字、关键词、标识符显示为符号。但拿到*整条*消息的人会连解码表一起拿到 |
| “重复你的指令” / 提示词注入 | **没有保证。** 会重复上下文的模型也会重复解码表。有些模型对“不要泄露这张表”比对“不要泄露你的提示词”更听话：请用 `evaluate_extraction` 实测 |
| 能导出完整上下文（含锚点）的攻击者 | **全部还原。** 用 `LeakDetector` 检测，不要依赖保密 |
| 提示词里的凭证和个人信息 | 不要混淆，直接删掉（`audit` 能找出来） |
| 掌握 seed 的内部人员 | 可复现别名；seed 并不是密码学意义上的密钥 |

## 常见问题

**模型真的会按符号执行吗？** 常常可以，但会因模型、提示词长度与强度而异，而且这里还没有用真实模型实测过。所以才提供 `evaluate_fidelity` 和 `evaluate_extraction`——请用它们作为上线门槛。
**值得用吗？** 如果你能把敏感逻辑挪出提示词（服务端工具、分类器、模型看不到的配置），就那样做。只有当提示词必须承载逻辑，而你想要一层针对随意暴露的防护外加泄露检测时，才用 ObfusAI。
**工具调用还能用吗？** 能。只混淆 `description` / `title` 文本，名称、参数、枚举与类型原样保留。
**为什么不同进程里同一个提示词长得不一样？** 默认 seed 每个实例随机。传入固定的 `seed` 即可得到可复现的输出（也便于利用供应商的提示词缓存）。
**`{占位符}` 会怎样？** 它们会被保留，你的模板渲染不受影响。

## 路线图

Chat Completions 的 `parse` / `stream` 辅助方法 · 混淆以 pydantic 类给出的结构化输出 Schema · 还原工具调用参数里的别名。

## 开发

```bash
pip install -e ".[dev]" && pytest
python examples/basic_usage.py && python examples/advanced_usage.py
```

## 许可证

MIT — 详见 [LICENSE](LICENSE)。

## ⭐ 支持 ObfusAI

如果这个项目帮你避免了一次提示词泄露，或者你只是喜欢这个想法：

- **[给仓库点个 Star](https://github.com/GANGMU-SBOM/ObfusAI)**——成本最低、帮助最大的支持方式
- 把它分享给正在上线 LLM 功能的同事
- 提个 [Issue](https://github.com/GANGMU-SBOM/ObfusAI/issues)：反馈 Bug、你需要的检测器，或它在哪个模型上表现不佳

[![Star History Chart](https://api.star-history.com/svg?repos=GANGMU-SBOM/ObfusAI&type=Date)](https://star-history.com/#GANGMU-SBOM/ObfusAI&Date)
