"""LlamaIndex adapters for ObfusAI.

Requires ``llama-index-core`` (``pip install obfusai[llamaindex]``).

* :class:`ObfusAILLM` - wraps any LlamaIndex ``LLM``.  System messages are
  obfuscated (one shared alias table per call, anchor attached to the first
  system message), and aliases in replies are decoded, also while streaming and
  for async calls.  Plain ``complete(prompt)`` calls are obfuscated as a whole::

      llm = ObfusAILLM(OpenAI(model="gpt-4o"), ObfusAIComponent("high", seed="..."))
      index.as_query_engine(llm=llm)

  Configure ``system_prompt`` on the wrapper, not on the wrapped LLM.

  The wrapper is a ``FunctionCallingLLM``: ``chat_with_tools``, ``predict_and_call``,
  agents and the streaming variants work when the wrapped LLM supports them.  Tool
  names, parameter names and enums are left untouched; the ``description`` / ``title``
  text of dict-based tool schemas (OpenAI, Anthropic, Mistral and similar providers)
  is obfuscated with the same alias table as the system prompt.  Providers that pass
  tool specs as non-dict objects are forwarded unchanged.  Aliases inside the model's
  tool-call arguments are not decoded.

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

import copy
from typing import Any, AsyncGenerator, Dict, Generator, List, Mapping, Optional, Sequence, Tuple, Union

try:
    from llama_index.core.base.llms.types import TextBlock
    from llama_index.core.bridge.pydantic import Field
    from llama_index.core.llms import (
        ChatMessage,
        ChatResponse,
        CompletionResponse,
        LLMMetadata,
        MessageRole,
    )
    from llama_index.core.llms.function_calling import FunctionCallingLLM
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

from ..core import ObfusAIComponent, _collect_schema_text
from ..guard import StreamRestorer

__all__ = ["ObfusAILLM", "ObfusAINodePostprocessor", "ObfusAITransform"]

_SYSTEM_ROLES = (MessageRole.SYSTEM, MessageRole.DEVELOPER)


# --------------------------------------------------------------------------- messages


def _protect_request(
    component: ObfusAIComponent,
    messages: Sequence[ChatMessage],
    roles: Sequence[MessageRole],
    kwargs: Mapping[str, Any],
) -> Tuple[List[ChatMessage], Dict[str, Any], Mapping[str, str]]:
    """Obfuscate ``roles`` messages and dict-based ``tools`` schemas with one alias table.

    Returns copies of the messages and kwargs (the inputs are never mutated) plus the table.
    """
    copies: List[ChatMessage] = [m.model_copy(deep=True) for m in messages]
    new_kwargs: Dict[str, Any] = dict(kwargs)
    blocks: List[Tuple[Any, str]] = []  # (text block, policy field = role name)
    first_target: Optional[ChatMessage] = None
    for message in copies:
        if message.role not in roles:
            continue
        field = message.role.value
        if first_target is None and not component.is_skipped(field):
            first_target = message
        blocks.extend((b, field) for b in message.blocks if getattr(b, "block_type", None) == "text")

    tool_slots: List[Tuple[Any, Any, str]] = []
    if isinstance(new_kwargs.get("tools"), list):
        new_kwargs["tools"] = copy.deepcopy(new_kwargs["tools"])
        _collect_schema_text(new_kwargs["tools"], ("description", "title"), tool_slots)

    session = component.session(
        [b.text for b, _ in blocks] + [container[key] for container, key, _ in tool_slots]
    )
    for block, field in blocks:
        block.text = session.encode(block.text, field)
    for container, key, field in tool_slots:
        container[key] = session.encode(container[key], field)
    anchor = session.anchor
    if not anchor:
        return copies, new_kwargs, {}

    if first_target is None:
        copies.insert(0, ChatMessage(role=MessageRole.SYSTEM, content=anchor))
    else:
        text_blocks = [b for b in first_target.blocks if getattr(b, "block_type", None) == "text"]
        if text_blocks:
            index = 0 if component.anchor_position == "prepend" else -1
            text_blocks[index].text = session.decorate(text_blocks[index].text)
        else:
            first_target.blocks.append(TextBlock(text=anchor))
    return copies, new_kwargs, session.mapping


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


class ObfusAILLM(FunctionCallingLLM):
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

    def _protect_chat(
        self, messages: Sequence[ChatMessage], kwargs: Mapping[str, Any]
    ) -> Tuple[List[ChatMessage], Dict[str, Any], Mapping[str, str]]:
        return _protect_request(self.component, messages, self.roles, kwargs)

    def _protect_prompt(self, prompt: str) -> Tuple[str, Mapping[str, str]]:
        if not self.protect_completions:
            return prompt, {}
        session = self.component.session([prompt])
        encoded = session.encode(prompt, "prompt")
        return session.decorate(encoded), session.mapping

    # -- chat

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        protected, kwargs, mapping = self._protect_chat(messages, kwargs)
        response = self.llm.chat(protected, **kwargs)
        return _restore_chat_response(response, mapping) if mapping and self.restore_responses else response

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        protected, kwargs, mapping = self._protect_chat(messages, kwargs)
        response = await self.llm.achat(protected, **kwargs)
        return _restore_chat_response(response, mapping) if mapping and self.restore_responses else response

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        protected, kwargs, mapping = self._protect_chat(messages, kwargs)
        inner = self.llm.stream_chat(protected, **kwargs)
        if not mapping or not self.restore_responses:
            return inner  # type: ignore[no-any-return]
        return _sync_stream(inner, _ChatStreamRestorer(mapping))

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        protected, kwargs, mapping = self._protect_chat(messages, kwargs)
        inner = await self.llm.astream_chat(protected, **kwargs)
        if not mapping or not self.restore_responses:
            return inner  # type: ignore[no-any-return]
        return _async_stream(inner, _ChatStreamRestorer(mapping))

    # -- function calling (delegated to the wrapped LLM; protection happens in chat/stream_chat)

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[Any],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not hasattr(self.llm, "_prepare_chat_with_tools_compat"):
            raise NotImplementedError(f"{type(self.llm).__name__} does not support tool calling")
        return self.llm._prepare_chat_with_tools_compat(  # type: ignore[no-any-return]
            tools,
            user_msg=user_msg,
            chat_history=chat_history,
            verbose=verbose,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            tool_required=tool_required,
            **kwargs,
        )

    def _validate_chat_with_tools_response(self, response: ChatResponse, tools: Sequence[Any], allow_parallel_tool_calls: bool = False, **kwargs: Any) -> ChatResponse:
        validate = getattr(self.llm, "_validate_chat_with_tools_response", None)
        if validate is None:
            return response
        return validate(response, tools, allow_parallel_tool_calls=allow_parallel_tool_calls, **kwargs)  # type: ignore[no-any-return]

    def get_tool_calls_from_response(self, response: ChatResponse, error_on_no_tool_call: bool = True, **kwargs: Any) -> List[Any]:
        if not hasattr(self.llm, "get_tool_calls_from_response"):
            raise NotImplementedError(f"{type(self.llm).__name__} does not support tool calling")
        return self.llm.get_tool_calls_from_response(  # type: ignore[no-any-return]
            response, error_on_no_tool_call=error_on_no_tool_call, **kwargs
        )

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
    clone.text = session.encode(clone.text, "nodes")
    for key in metadata_keys:
        clone.metadata[key] = session.encode(clone.metadata[key], "nodes")
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
