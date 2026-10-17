"""LlamaIndex adapters for ObfusAI.

Requires ``llama-index-core`` (``pip install obfusai[llamaindex]``).

* :class:`ObfusAILLM` - wraps any LlamaIndex ``LLM``.  System messages are
  obfuscated (one shared alias table per call, anchor attached to the first
  system message), and aliases in replies are decoded, also while streaming and
  for async calls.  Plain ``complete(prompt)`` calls are obfuscated as a whole::

      llm = ObfusAILLM(OpenAI(model="gpt-4o"), ObfusAIComponent("high", seed="..."))
      index.as_query_engine(llm=llm)

  Configure ``system_prompt`` on the wrapper, not on the wrapped LLM.  Tool
  calling (``FunctionCallingLLM`` methods) is not forwarded.

* :class:`ObfusAINodePostprocessor` - obfuscates retrieved nodes right before they
  are placed into the prompt, so retrieval itself still runs on readable text::

      query_engine = index.as_query_engine(node_postprocessors=[ObfusAINodePostprocessor()])

* :class:`ObfusAITransform` - an ingestion ``TransformComponent`` for pipelines that
  store obfuscated text.  Embeddings of obfuscated text retrieve poorly; prefer the
  postprocessor unless that is what you want.

Each node carries its own alias table and anchor, so any subset of retrieved nodes
stays decodable.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Generator, List, Mapping, Optional, Sequence, Tuple

try:
    from llama_index.core.base.llms.types import TextBlock
    from llama_index.core.bridge.pydantic import Field
    from llama_index.core.llms import (
        LLM,
        ChatMessage,
        ChatResponse,
        CompletionResponse,
        LLMMetadata,
        MessageRole,
    )
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import (
        BaseNode,
        NodeWithScore,
        QueryBundle,
        TextNode,
        TransformComponent,
    )
except ImportError as exc:  # pragma: no cover - exercised only without llama-index-core
    raise ImportError(
        "obfusai.adapters.llamaindex requires llama-index-core; "
        "install it with `pip install obfusai[llamaindex]`."
    ) from exc

from ..core import ObfusAIComponent
from ..guard import StreamRestorer

__all__ = ["ObfusAILLM", "ObfusAINodePostprocessor", "ObfusAITransform"]

_SYSTEM_ROLES = (MessageRole.SYSTEM, MessageRole.DEVELOPER)


# --------------------------------------------------------------------------- messages


def _protect_messages(
    component: ObfusAIComponent, messages: Sequence[ChatMessage], roles: Sequence[MessageRole]
) -> Tuple[List[ChatMessage], Mapping[str, str]]:
    """Obfuscate the text of ``roles`` messages with one table; returns copies and the table."""
    copies: List[ChatMessage] = [m.model_copy(deep=True) for m in messages]
    blocks: List[Any] = []
    first_target: Optional[ChatMessage] = None
    for message in copies:
        if message.role not in roles:
            continue
        first_target = first_target or message
        blocks.extend(b for b in message.blocks if getattr(b, "block_type", None) == "text")

    session = component.session(b.text for b in blocks)
    for block in blocks:
        block.text = session.encode(block.text)
    anchor = session.anchor
    if not anchor:
        return copies, {}

    if first_target is None:
        copies.insert(0, ChatMessage(role=MessageRole.SYSTEM, content=anchor))
    else:
        text_blocks = [b for b in first_target.blocks if getattr(b, "block_type", None) == "text"]
        if text_blocks:
            index = 0 if component.anchor_position == "prepend" else -1
            text_blocks[index].text = session.decorate(text_blocks[index].text)
        else:
            first_target.blocks.append(TextBlock(text=anchor))
    return copies, session.mapping


def _restore_chat_response(response: ChatResponse, mapping: Mapping[str, str]) -> ChatResponse:
    for block in response.message.blocks:
        if getattr(block, "block_type", None) == "text":
            block.text = ObfusAIComponent.deobfuscate(block.text, mapping)
    if response.delta:
        response.delta = ObfusAIComponent.deobfuscate(response.delta, mapping)
    return response


def _restore_completion(response: CompletionResponse, mapping: Mapping[str, str]) -> CompletionResponse:
    response.text = ObfusAIComponent.deobfuscate(response.text, mapping)
    if response.delta:
        response.delta = ObfusAIComponent.deobfuscate(response.delta, mapping)
    return response


# --------------------------------------------------------------------------- streaming


class _ChatStreamRestorer:
    """Decode a stream of cumulative ``ChatResponse`` chunks, flushing the held-back tail at the end."""

    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._restorer = StreamRestorer(mapping)
        self._seen_raw = ""
        self._text = ""
        self._last: Optional[ChatResponse] = None

    def process(self, chunk: ChatResponse) -> ChatResponse:
        raw_total = chunk.message.content or ""
        delta = chunk.delta if chunk.delta is not None else raw_total[len(self._seen_raw) :]
        self._seen_raw = raw_total or self._seen_raw + delta
        piece = self._restorer.feed(delta)
        self._text += piece
        out = self._emit(chunk, piece)
        self._last = out
        return out

    def finish(self) -> List[ChatResponse]:
        tail = self._restorer.flush()
        if not tail or self._last is None:
            return []
        self._text += tail
        return [self._emit(self._last, tail)]

    def _emit(self, template: ChatResponse, delta: str) -> ChatResponse:
        out = template.model_copy()
        out.message = template.message.model_copy(deep=True)
        out.message.content = self._text
        out.delta = delta
        return out


class _CompletionStreamRestorer:
    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._restorer = StreamRestorer(mapping)
        self._seen_raw = ""
        self._text = ""
        self._last: Optional[CompletionResponse] = None

    def process(self, chunk: CompletionResponse) -> CompletionResponse:
        raw_total = chunk.text or ""
        delta = chunk.delta if chunk.delta is not None else raw_total[len(self._seen_raw) :]
        self._seen_raw = raw_total or self._seen_raw + delta
        piece = self._restorer.feed(delta)
        self._text += piece
        out = self._emit(chunk, piece)
        self._last = out
        return out

    def finish(self) -> List[CompletionResponse]:
        tail = self._restorer.flush()
        if not tail or self._last is None:
            return []
        self._text += tail
        return [self._emit(self._last, tail)]

    def _emit(self, template: CompletionResponse, delta: str) -> CompletionResponse:
        out = template.model_copy()
        out.text = self._text
        out.delta = delta
        return out


def _sync_stream(inner: Any, restorer: Any) -> Generator[Any, None, None]:
    for chunk in inner:
        yield restorer.process(chunk)
    yield from restorer.finish()


async def _async_stream(inner: Any, restorer: Any) -> AsyncGenerator[Any, None]:
    async for chunk in inner:
        yield restorer.process(chunk)
    for out in restorer.finish():
        yield out


# --------------------------------------------------------------------------- LLM wrapper


class ObfusAILLM(LLM):
    """Protect any LlamaIndex ``LLM``: obfuscate system prompts, decode replies.

    Args:
        llm: The wrapped LlamaIndex LLM.
        component: Engine to use (defaults to ``ObfusAIComponent()``).
        roles: Message roles whose text is obfuscated (default system and developer).
        protect_completions: Obfuscate whole ``complete``/``stream_complete`` prompts
            (default ``True``).  With ``False`` those calls are sent unchanged.
        restore_responses: Decode aliases found in replies (default ``True``).
    """

    llm: Any = Field(description="The wrapped LLM.")
    component: Any = Field(description="The ObfusAIComponent used for protection.")
    roles: Tuple[MessageRole, ...] = Field(default=_SYSTEM_ROLES)
    protect_completions: bool = True
    restore_responses: bool = True

    def __init__(
        self,
        llm: Any,
        component: Optional[ObfusAIComponent] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, component=component or ObfusAIComponent(), **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "ObfusAILLM"

    @property
    def metadata(self) -> LLMMetadata:
        return self.llm.metadata

    # -- helpers

    def _protect_chat(self, messages: Sequence[ChatMessage]) -> Tuple[List[ChatMessage], Mapping[str, str]]:
        return _protect_messages(self.component, messages, self.roles)

    def _protect_prompt(self, prompt: str) -> Tuple[str, Mapping[str, str]]:
        if not self.protect_completions:
            return prompt, {}
        session = self.component.session([prompt])
        encoded = session.encode(prompt)
        return session.decorate(encoded), session.mapping

    # -- chat

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        protected, mapping = self._protect_chat(messages)
        response = self.llm.chat(protected, **kwargs)
        return _restore_chat_response(response, mapping) if mapping and self.restore_responses else response

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        protected, mapping = self._protect_chat(messages)
        response = await self.llm.achat(protected, **kwargs)
        return _restore_chat_response(response, mapping) if mapping and self.restore_responses else response

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        protected, mapping = self._protect_chat(messages)
        inner = self.llm.stream_chat(protected, **kwargs)
        if not mapping or not self.restore_responses:
            return inner  # type: ignore[no-any-return]
        return _sync_stream(inner, _ChatStreamRestorer(mapping))

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        protected, mapping = self._protect_chat(messages)
        inner = await self.llm.astream_chat(protected, **kwargs)
        if not mapping or not self.restore_responses:
            return inner  # type: ignore[no-any-return]
        return _async_stream(inner, _ChatStreamRestorer(mapping))

    # -- completion

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        text, mapping = self._protect_prompt(prompt)
        response = self.llm.complete(text, formatted=formatted, **kwargs)
        return _restore_completion(response, mapping) if mapping and self.restore_responses else response

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        text, mapping = self._protect_prompt(prompt)
        response = await self.llm.acomplete(text, formatted=formatted, **kwargs)
        return _restore_completion(response, mapping) if mapping and self.restore_responses else response

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        text, mapping = self._protect_prompt(prompt)
        inner = self.llm.stream_complete(text, formatted=formatted, **kwargs)
        if not mapping or not self.restore_responses:
            return inner  # type: ignore[no-any-return]
        return _sync_stream(inner, _CompletionStreamRestorer(mapping))

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        text, mapping = self._protect_prompt(prompt)
        inner = await self.llm.astream_complete(text, formatted=formatted, **kwargs)
        if not mapping or not self.restore_responses:
            return inner  # type: ignore[no-any-return]
        return _async_stream(inner, _CompletionStreamRestorer(mapping))


# --------------------------------------------------------------------------- nodes


def _obfuscate_node(
    node: BaseNode, component: ObfusAIComponent, include_anchor: bool, obfuscate_metadata: bool
) -> BaseNode:
    """Return a copy of ``node`` whose text (and optionally string metadata) is obfuscated."""
    if not isinstance(node, TextNode):
        return node
    clone = node.model_copy(deep=True)
    metadata_keys = (
        [k for k, v in clone.metadata.items() if isinstance(v, str)] if obfuscate_metadata else []
    )
    session = component.session([clone.text] + [clone.metadata[k] for k in metadata_keys])
    clone.text = session.encode(clone.text)
    for key in metadata_keys:
        clone.metadata[key] = session.encode(clone.metadata[key])
    if include_anchor:
        clone.text = session.decorate(clone.text)
    else:
        clone.metadata["obfusai_anchor"] = session.anchor
        clone.excluded_embed_metadata_keys = list(clone.excluded_embed_metadata_keys) + ["obfusai_anchor"]
    return clone


class ObfusAINodePostprocessor(BaseNodePostprocessor):
    """Obfuscate retrieved nodes before they enter the LLM prompt.

    Args:
        component: Engine to use (defaults to ``ObfusAIComponent()``).
        include_anchor: Append each node's anchor to its text (default ``True``).  When
            ``False`` the anchor is stored in ``metadata['obfusai_anchor']`` for you to place.
        obfuscate_metadata: Also obfuscate string metadata values (they are part of the prompt).
    """

    component: Any = Field(default=None, description="ObfusAIComponent")
    include_anchor: bool = True
    obfuscate_metadata: bool = False

    def __init__(self, component: Optional[ObfusAIComponent] = None, **kwargs: Any) -> None:
        super().__init__(component=component or ObfusAIComponent(), **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "ObfusAINodePostprocessor"

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        return [
            NodeWithScore(
                node=_obfuscate_node(n.node, self.component, self.include_anchor, self.obfuscate_metadata),
                score=n.score,
            )
            for n in nodes
        ]


class ObfusAITransform(TransformComponent):
    """Ingestion transform that obfuscates node text (see module docs for the retrieval trade-off)."""

    component: Any = Field(default=None, description="ObfusAIComponent")
    include_anchor: bool = True
    obfuscate_metadata: bool = False

    def __init__(self, component: Optional[ObfusAIComponent] = None, **kwargs: Any) -> None:
        super().__init__(component=component or ObfusAIComponent(), **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "ObfusAITransform"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[BaseNode]:
        return [
            _obfuscate_node(n, self.component, self.include_anchor, self.obfuscate_metadata)
            for n in nodes
        ]
