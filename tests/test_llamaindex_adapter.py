from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.llms import (  # noqa: E402
    LLM,
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode  # noqa: E402

from obfusai import ObfusAIComponent  # noqa: E402
from obfusai.adapters.llamaindex import (  # noqa: E402
    ObfusAILLM,
    ObfusAINodePostprocessor,
    ObfusAITransform,
)
from obfusai.anchor import decode_with_anchor, strip_anchors  # noqa: E402

SYSTEM = "Rule: BUY when edge_bps > 35. Codename project_falcon."


def _alias_of(text: str) -> str:
    return text.split(" > ")[0].split()[-1]


class FakeLLM(LLM):
    """Echoes the first alias it finds in the system prompt/prompt, like a model quoting a symbol."""

    seen: List[Any] = []
    chunk: int = 3

    @classmethod
    def class_name(cls) -> str:
        return "FakeLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_chat_model=True)

    def _reply(self, text: str) -> str:
        return f"The metric {_alias_of(text)} matters."

    def _pieces(self, text: str) -> List[str]:
        return [text[i : i + self.chunk] for i in range(0, len(text), self.chunk)]

    def chat(self, messages, **kwargs):
        self.seen.append(messages)
        text = self._reply(messages[0].content)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text))

    async def achat(self, messages, **kwargs):
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages, **kwargs):
        self.seen.append(messages)
        text, acc = self._reply(messages[0].content), ""
        for piece in self._pieces(text):
            acc += piece
            yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=acc), delta=piece)

    async def astream_chat(self, messages, **kwargs):
        async def gen():
            for r in self.stream_chat(messages, **kwargs):
                yield r
        return gen()

    def complete(self, prompt, formatted=False, **kwargs):
        self.seen.append(prompt)
        return CompletionResponse(text=self._reply(prompt))

    async def acomplete(self, prompt, formatted=False, **kwargs):
        return self.complete(prompt, formatted=formatted, **kwargs)

    def stream_complete(self, prompt, formatted=False, **kwargs):
        self.seen.append(prompt)
        text, acc = self._reply(prompt), ""
        for piece in self._pieces(text):
            acc += piece
            yield CompletionResponse(text=acc, delta=piece)

    async def astream_complete(self, prompt, formatted=False, **kwargs):
        async def gen():
            for r in self.stream_complete(prompt, formatted=formatted, **kwargs):
                yield r
        return gen()


EXPECTED = "The metric edge_bps matters."


def _wrap(**opts: Any) -> tuple:
    inner = FakeLLM(seen=[])
    return inner, ObfusAILLM(inner, ObfusAIComponent("low", seed="s"), **opts)


def _messages() -> List[ChatMessage]:
    return [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM), ChatMessage(role=MessageRole.USER, content="hi")]


def test_chat_protects_system_and_restores_reply() -> None:
    inner, llm = _wrap()
    original = _messages()
    reply = llm.chat(original)
    sent = inner.seen[0]
    assert decode_with_anchor(sent[0].content) == SYSTEM
    assert "edge_bps" not in sent[0].content.split("<obfusai-context")[0]
    assert sent[1].content == "hi"
    assert original[0].content == SYSTEM  # caller's messages untouched
    assert reply.message.content == EXPECTED
    assert llm.metadata.is_chat_model


def test_chat_without_system_text_adds_nothing_and_missing_system_gets_one() -> None:
    inner, llm = _wrap()
    llm.chat([ChatMessage(role=MessageRole.SYSTEM, content="be kind"), ChatMessage(role=MessageRole.USER, content="x")])
    assert inner.seen[0][0].content == "be kind"
    assert len(inner.seen[0]) == 2


@pytest.mark.parametrize("chunk", [1, 2, 3, 100])
def test_stream_chat_restores_incrementally(chunk: int) -> None:
    inner = FakeLLM(seen=[], chunk=chunk)
    llm = ObfusAILLM(inner, ObfusAIComponent("high", seed="s"))
    responses = list(llm.stream_chat(_messages()))
    assert "".join(r.delta for r in responses) == EXPECTED
    assert responses[-1].message.content == EXPECTED
    assert all(EXPECTED.startswith(r.message.content) for r in responses)  # cumulative text only ever grows


def test_complete_paths() -> None:
    inner, llm = _wrap()
    reply = llm.complete(SYSTEM)
    assert reply.text == EXPECTED and "<obfusai-context" in inner.seen[0]
    assert "".join(r.delta for r in llm.stream_complete(SYSTEM)) == EXPECTED
    assert list(llm.stream_complete(SYSTEM))[-1].text == EXPECTED
    raw_inner, raw = _wrap(protect_completions=False)
    raw.complete(SYSTEM)
    assert raw_inner.seen[0] == SYSTEM


def test_async_paths() -> None:
    async def run() -> tuple:
        inner, llm = _wrap()
        chat = await llm.achat(_messages())
        comp = await llm.acomplete(SYSTEM)
        chat_stream = await llm.astream_chat(_messages())
        chat_text = "".join([r.delta async for r in chat_stream])
        comp_stream = await llm.astream_complete(SYSTEM)
        comp_text = "".join([r.delta async for r in comp_stream])
        return chat.message.content, comp.text, chat_text, comp_text

    assert asyncio.run(run()) == (EXPECTED,) * 4


def test_restore_can_be_disabled() -> None:
    _, llm = _wrap(restore_responses=False)
    assert llm.chat(_messages()).message.content != EXPECTED


def test_predict_uses_wrapper_system_prompt() -> None:
    from llama_index.core.prompts import PromptTemplate

    inner = FakeLLM(seen=[])
    llm = ObfusAILLM(inner, ObfusAIComponent("low", seed="s"), system_prompt=SYSTEM)
    out = llm.predict(PromptTemplate("Question: {q}"), q="why")
    assert out == EXPECTED
    assert decode_with_anchor(inner.seen[0][0].content) == SYSTEM


# ----------------------------------------------------------------- nodes
def _nodes() -> List[NodeWithScore]:
    node = TextNode(text="Alert when edge_bps > 35.5 for project_falcon.", metadata={"owner": "desk_alpha", "page": 3})
    return [NodeWithScore(node=node, score=0.9)]


def test_node_postprocessor() -> None:
    nodes = _nodes()
    out = ObfusAINodePostprocessor(ObfusAIComponent("low", seed="s")).postprocess_nodes(nodes, QueryBundle("q"))
    text = out[0].node.text
    assert "edge_bps" not in text.split("<obfusai-context")[0]
    assert decode_with_anchor(text) == nodes[0].node.text
    assert out[0].score == 0.9
    assert nodes[0].node.text.startswith("Alert")  # original node untouched
    assert out[0].node.metadata["owner"] == "desk_alpha"


def test_node_postprocessor_metadata_and_no_anchor() -> None:
    post = ObfusAINodePostprocessor(
        ObfusAIComponent("low", seed="s"), include_anchor=False, obfuscate_metadata=True
    )
    node = post.postprocess_nodes(_nodes(), QueryBundle("q"))[0].node
    assert "desk_alpha" not in node.metadata["owner"] and node.metadata["page"] == 3
    assert "<obfusai-context" not in node.text and "<obfusai-context" in node.metadata["obfusai_anchor"]
    assert "obfusai_anchor" in node.excluded_embed_metadata_keys


def test_transform_component() -> None:
    nodes = [n.node for n in _nodes()] + [TextNode(text="")]
    out = ObfusAITransform(ObfusAIComponent("low", seed="s"))(nodes)
    assert "edge_bps" not in out[0].text.split("<obfusai-context")[0] and out[1].text == ""


def test_query_engine_end_to_end() -> None:
    from llama_index.core import Document, SummaryIndex

    inner = FakeLLM(seen=[])
    llm = ObfusAILLM(inner, ObfusAIComponent("low", seed="s"), system_prompt=SYSTEM)
    index = SummaryIndex.from_documents([Document(text="Alert when edge_bps > 35.5 for project_falcon.")])
    engine = index.as_query_engine(llm=llm, node_postprocessors=[ObfusAINodePostprocessor(ObfusAIComponent("low", seed="n"))])
    response = engine.query("what matters?")
    assert EXPECTED in str(response)
    assert inner.seen, "the wrapped LLM was never called"
    for call in inner.seen:
        for message in call:
            visible = strip_anchors(str(message.content))
            assert "edge_bps" not in visible and "project_falcon" not in visible


def test_policies_apply_to_roles_and_nodes() -> None:
    from obfusai import FieldPolicy

    engine = ObfusAIComponent("low", seed="s", policies={"system": FieldPolicy(skip=True), "nodes": FieldPolicy(skip=True)})
    inner = FakeLLM(seen=[])
    ObfusAILLM(inner, engine).chat(_messages())
    assert inner.seen[0][0].content == SYSTEM  # system skipped -> sent untouched
    node = ObfusAINodePostprocessor(engine).postprocess_nodes(_nodes(), QueryBundle("q"))[0].node
    assert node.text == _nodes()[0].node.text


# ----------------------------------------------------------------- function calling / tool schemas
def _tool_llm_class():
    from llama_index.core.llms.function_calling import FunctionCallingLLM
    from llama_index.core.tools import ToolSelection

    class FakeToolLLM(FunctionCallingLLM):
        """OpenAI-style tool plumbing: tool specs travel as dicts in the `tools` kwarg."""

        seen: List[Any] = []

        @classmethod
        def class_name(cls) -> str:
            return "FakeToolLLM"

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(is_chat_model=True, is_function_calling_model=True)

        def _prepare_chat_with_tools(self, tools, user_msg=None, chat_history=None, verbose=False,
                                     allow_parallel_tool_calls=False, tool_required=False, **kwargs):
            history = list(chat_history or [])
            if isinstance(user_msg, str):
                user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
            if user_msg is not None:
                history.append(user_msg)
            specs = [
                {"type": "function", "function": {"name": t.metadata.name, "description": t.metadata.description,
                                                  "parameters": t.metadata.get_parameters_dict()}}
                for t in tools
            ]
            return {"messages": history, "tools": specs}

        def get_tool_calls_from_response(self, response, error_on_no_tool_call=True, **kwargs):
            return [ToolSelection(tool_id="1", tool_name="get_signal", tool_kwargs={"ticker": "NVDA"})]

        def chat(self, messages, **kwargs):
            self.seen.append((messages, kwargs))
            sys_text = next((m.content for m in messages if m.role == MessageRole.SYSTEM), "")
            alias = _alias_of(sys_text) if " > " in sys_text else "none"
            return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=f"Using {alias} now."))

        async def achat(self, messages, **kwargs):
            return self.chat(messages, **kwargs)

        def stream_chat(self, messages, **kwargs):
            self.seen.append((messages, kwargs))
            text, acc = "Using edge_bps now.", ""
            for piece in [text[i : i + 4] for i in range(0, len(text), 4)]:
                acc += piece
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=acc), delta=piece)

        async def astream_chat(self, messages, **kwargs):
            async def gen():
                for r in self.stream_chat(messages, **kwargs):
                    yield r
            return gen()

        def complete(self, prompt, formatted=False, **kwargs):
            return CompletionResponse(text="x")

        async def acomplete(self, prompt, formatted=False, **kwargs):
            return self.complete(prompt)

        def stream_complete(self, prompt, formatted=False, **kwargs):
            yield CompletionResponse(text="x", delta="x")

        async def astream_complete(self, prompt, formatted=False, **kwargs):
            async def gen():
                yield CompletionResponse(text="x", delta="x")
            return gen()

    return FakeToolLLM


def _signal_tool():
    from llama_index.core.tools import FunctionTool

    def get_signal(ticker: str) -> str:
        return f"{ticker}: BUY"

    return FunctionTool.from_defaults(fn=get_signal, name="get_signal", description="Reads edge_bps (limit 42.5) for a ticker.")


def test_chat_with_tools_obfuscates_tool_schema_and_system_prompt() -> None:
    Fake = _tool_llm_class()
    inner = Fake(seen=[])
    llm = ObfusAILLM(inner, ObfusAIComponent("low", seed="s"), system_prompt=None)
    tool = _signal_tool()
    resp = llm.chat_with_tools(
        [tool],
        chat_history=[ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM)],
        user_msg="go",
    )
    messages, kwargs = inner.seen[0]
    spec = kwargs["tools"][0]["function"]
    assert spec["name"] == "get_signal"
    assert "edge_bps" not in spec["description"] and "42.5" not in spec["description"]
    assert set(spec["parameters"]["properties"]) == {"ticker"}  # parameter names untouched
    assert decode_with_anchor(messages[0].content) == SYSTEM  # one anchor decodes both prompt and tool text
    assert "edge_bps" in decode_with_anchor("\n".join([messages[0].content, spec["description"]]))
    assert resp.message.content == "Using edge_bps now."
    assert tool.metadata.description == "Reads edge_bps (limit 42.5) for a ticker."  # caller's tool untouched


def test_tool_only_request_gets_a_dedicated_anchor_message() -> None:
    Fake = _tool_llm_class()
    inner = Fake(seen=[])
    llm = ObfusAILLM(inner, ObfusAIComponent("low", seed="s"))
    llm.chat_with_tools([_signal_tool()], user_msg="go")
    messages, kwargs = inner.seen[0]
    assert messages[0].role == MessageRole.SYSTEM and messages[0].content.startswith("<obfusai-context")
    assert "edge_bps" not in kwargs["tools"][0]["function"]["description"]


def test_predict_and_call_and_agent_metadata() -> None:
    Fake = _tool_llm_class()
    llm = ObfusAILLM(Fake(seen=[]), ObfusAIComponent("low", seed="s"))
    assert llm.metadata.is_function_calling_model
    out = llm.predict_and_call([_signal_tool()], "go")
    assert "NVDA: BUY" in str(out)


def test_tool_streaming_and_async_variants() -> None:
    Fake = _tool_llm_class()
    llm = ObfusAILLM(Fake(seen=[]), ObfusAIComponent("low", seed="s"))
    history = [ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM)]
    text = "".join(r.delta for r in llm.stream_chat_with_tools([_signal_tool()], chat_history=history, user_msg="go"))
    assert text == "Using edge_bps now."

    async def run() -> tuple:
        chat = await llm.achat_with_tools([_signal_tool()], chat_history=history, user_msg="go")
        stream = await llm.astream_chat_with_tools([_signal_tool()], chat_history=history, user_msg="go")
        return chat.message.content, "".join([r.delta async for r in stream])

    assert asyncio.run(run()) == ("Using edge_bps now.", "Using edge_bps now.")


def test_tools_policy_skip_leaves_schema_readable() -> None:
    from obfusai import FieldPolicy

    Fake = _tool_llm_class()
    inner = Fake(seen=[])
    engine = ObfusAIComponent("low", seed="s", policies={"tools": FieldPolicy(skip=True)})
    ObfusAILLM(inner, engine).chat_with_tools([_signal_tool()], chat_history=[ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM)], user_msg="go")
    assert "edge_bps" in inner.seen[0][1]["tools"][0]["function"]["description"]


def test_non_function_calling_inner_raises_clearly() -> None:
    inner = FakeLLM(seen=[])
    llm = ObfusAILLM(inner, ObfusAIComponent("low", seed="s"))
    with pytest.raises(NotImplementedError):
        llm.chat_with_tools([_signal_tool()], user_msg="go")
