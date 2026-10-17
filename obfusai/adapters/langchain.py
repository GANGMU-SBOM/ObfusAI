"""LangChain adapters for ObfusAI.

Requires ``langchain-core`` (``pip install obfusai[langchain]``).

* :class:`ObfusAIDocumentTransformer` - a ``BaseDocumentTransformer`` that
  obfuscates the ``page_content`` of documents (e.g. proprietary playbooks
  loaded for RAG).
* :class:`ObfusAIRunnable` - an LCEL ``Runnable`` placed between a prompt and a
  chat model.  It obfuscates system messages and attaches one shared anchor::

      chain = prompt | ObfusAIRunnable(ObfusAIComponent("high")) | chat_model
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

try:
    from langchain_core.documents import BaseDocumentTransformer, Document
    from langchain_core.messages import (
        BaseMessage,
        SystemMessage,
        convert_to_messages,
    )
    from langchain_core.prompt_values import PromptValue
    from langchain_core.runnables import Runnable, RunnableConfig
except ImportError as exc:  # pragma: no cover - exercised only without langchain-core
    raise ImportError(
        "obfusai.adapters.langchain requires langchain-core; "
        "install it with `pip install obfusai[langchain]`."
    ) from exc

from ..core import ObfusAIComponent

__all__ = ["ObfusAIDocumentTransformer", "ObfusAIRunnable"]

PromptInput = Union[str, PromptValue, Sequence[Any]]


class ObfusAIDocumentTransformer(BaseDocumentTransformer):
    """Obfuscate the content of each document independently.

    Every document gets its own alias table (so any subset of documents stays
    decodable when retrieved on its own).  With ``include_anchor=True`` the
    anchor is attached to ``page_content``; otherwise it is only stored in
    ``metadata['obfusai_anchor']`` for the caller to place in the prompt.
    Original values are never written to metadata.
    """

    def __init__(self, component: Optional[ObfusAIComponent] = None, *, include_anchor: bool = True) -> None:
        self.component = component or ObfusAIComponent()
        self.include_anchor = include_anchor

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        transformed: List[Document] = []
        for doc in documents:
            result = self.component.obfuscate(doc.page_content)
            metadata = dict(doc.metadata)
            metadata["obfusai_session_id"] = result.session_id
            if self.include_anchor:
                content = result.prompt
            else:
                content = result.text
                metadata["obfusai_anchor"] = result.anchor
            transformed.append(Document(page_content=content, metadata=metadata))
        return transformed


class ObfusAIRunnable(Runnable):  # type: ignore[type-arg]
    """LCEL step that obfuscates system instructions before they reach the model.

    Accepts a string, a ``PromptValue`` or a sequence of messages and returns a
    list of messages.  Messages whose type is in ``message_types`` have their
    text obfuscated with one shared alias table, and the anchor is appended to
    the first of them (a new ``SystemMessage`` is inserted if there is none).
    A bare string input is treated as a system prompt: the output is a
    ``[SystemMessage]`` list.  Call :meth:`restore` on the model's reply to map
    any aliases back to the original text.
    """

    def __init__(
        self,
        component: Optional[ObfusAIComponent] = None,
        *,
        message_types: Sequence[str] = ("system",),
    ) -> None:
        self.component = component or ObfusAIComponent()
        self.message_types = tuple(message_types)
        self._last_mapping: dict = {}

    @property
    def last_mapping(self) -> dict:
        """Alias table of the most recent ``invoke`` (not thread-safe; prefer ``invoke_with_mapping``)."""
        return dict(self._last_mapping)

    def restore(self, text: str) -> str:
        """Decode aliases in ``text`` using the most recent invocation's table."""
        return ObfusAIComponent.deobfuscate(text, self._last_mapping)

    def invoke(self, input: PromptInput, config: Optional[RunnableConfig] = None, **kwargs: Any) -> List[BaseMessage]:
        messages, _ = self.invoke_with_mapping(input)
        return messages

    def invoke_with_mapping(self, input: PromptInput) -> Tuple[List[BaseMessage], dict]:
        """Like :meth:`invoke` but also returns the alias table for this call."""
        if isinstance(input, str):
            messages: List[BaseMessage] = [SystemMessage(content=input)]
        elif isinstance(input, PromptValue):
            messages = list(input.to_messages())
        else:
            messages = list(convert_to_messages(input))

        slots: List[Tuple[int, Optional[int]]] = []  # (message index, content-part index or None)
        for i, message in enumerate(messages):
            if message.type not in self.message_types:
                continue
            if isinstance(message.content, str):
                slots.append((i, None))
            elif isinstance(message.content, list):
                for j, part in enumerate(message.content):
                    if isinstance(part, str) or (isinstance(part, dict) and isinstance(part.get("text"), str)):
                        slots.append((i, j))

        def read(i: int, j: Optional[int]) -> str:
            content = messages[i].content
            if j is None:
                return str(content)
            part = content[j]  # type: ignore[index]
            return part if isinstance(part, str) else part["text"]

        session = self.component.session(read(i, j) for i, j in slots)
        new_content = {}
        for i, j in slots:
            new_content[(i, j)] = session.encode(read(i, j))
        anchor = session.anchor

        updated: List[BaseMessage] = []
        anchored = not anchor
        for i, message in enumerate(messages):
            own = [(k, v) for k, v in new_content.items() if k[0] == i]
            if not own:
                updated.append(message)
                continue
            if isinstance(message.content, str):
                text = own[0][1]
                if not anchored:
                    text = session.decorate(text)
                    anchored = True
                updated.append(_with_content(message, text))
            else:
                parts = list(message.content)
                for (_, j), text in own:
                    part = parts[j]  # type: ignore[index]
                    parts[j] = text if isinstance(part, str) else {**part, "text": text}
                if not anchored:
                    parts.append({"type": "text", "text": anchor})
                    anchored = True
                updated.append(_with_content(message, parts))

        if not anchored:
            updated.insert(0, SystemMessage(content=anchor))

        mapping = session.mapping
        self._last_mapping = mapping
        return updated, mapping


def _with_content(message: BaseMessage, content: Any) -> BaseMessage:
    """Copy ``message`` with new content across pydantic v1/v2 based langchain-core releases."""
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    return message.copy(update={"content": content})  # pragma: no cover - legacy langchain-core
