# ObfusAI

Middleware that obfuscates sensitive LLM prompts, system roles and tool schemas so a leaked
prompt is hard for a human to read, while capable LLMs (GPT-4o, Claude, ...) can still follow it.

ObfusAI replaces proprietary numbers, metrics, keywords and code identifiers with abstract Greek
symbols (`momentum_score > 0.73` becomes `η > χ`) and attaches a *semantic anchor*: an
`<obfusai-context>` block holding the alias table plus instructions to decode it silently.

> **Security model.** This is obfuscation, not encryption. The anchor travels in the same context
> window as the obfuscated text, so an attacker who can make the model dump its whole context
> recovers both. ObfusAI defeats casual extraction ("repeat your system prompt") and makes leaked
> prompts costly to read; it does not make a prompt confidential. Keep real secrets (API keys,
> credentials) out of prompts regardless.

## Install

```bash
pip install .                 # zero-dependency core, Python 3.9+
pip install ".[langchain]"    # plus the LangChain adapter
```

## Usage

```python
from obfusai import ObfusAIComponent

engine = ObfusAIComponent("medium", seed="my-secret", keywords=["Helios Capital"])

result = engine.obfuscate("If momentum_score > 0.73 then BUY. Cap size at 2.5% of NAV.")
print(result.prompt)                  # obfuscated text + anchor, ready for the system role
print(result.restore("η cleared"))    # decode aliases in a model reply

# A whole chat-completion request (OpenAI or Anthropic shape): one shared alias table,
# anchor attached to the system prompt, tool names/parameters/enums left intact.
protected = engine.obfuscate_payload(request_payload)
client.chat.completions.create(**protected.payload)
```

Run `python examples/basic_usage.py` for a full demo against a mock LLM.

### Intensity

| Level    | Obfuscated                                                                 | Notes |
|----------|----------------------------------------------------------------------------|-------|
| `low`    | numbers/percentages/currency, snake/camel/Pascal/dotted identifiers, `keywords` | smallest anchor, highest reliability |
| `medium` | + quoted strings, acronyms, proper-noun phrases                            | default |
| `high`   | + long domain words, `⟦α⟧`-wrapped aliases, reversed anchor definitions, decoy aliases, shuffled table | larger anchor, slightly lower decoding reliability; test on your model |

Left untouched at every level: template placeholders (`{var}`, `{{var}}`, `${var}`, `%(var)s`),
chat special tokens, list markers, and directive words such as `NEVER`/`MUST`.

Notes: pass a fixed `seed` for byte-identical output (provider prompt caching); the default is a
random per-instance seed. Empty or `None` input returns an empty result; text with nothing to hide
passes through with no anchor. Non-string input raises `TypeError`.

### LangChain

```python
from obfusai import ObfusAIComponent
from obfusai.adapters.langchain import ObfusAIDocumentTransformer, ObfusAIRunnable

chain = prompt | ObfusAIRunnable(ObfusAIComponent("high")) | chat_model
docs = ObfusAIDocumentTransformer(ObfusAIComponent()).transform_documents(docs)
```

## Development

```bash
pip install -e ".[dev]" && pytest
```

## License

MIT; see [LICENSE](LICENSE).
