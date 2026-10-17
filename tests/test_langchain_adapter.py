from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from obfusai import ObfusAIComponent  # noqa: E402
from obfusai.adapters.langchain import ObfusAIDocumentTransformer, ObfusAIRunnable  # noqa: E402


def test_document_transformer() -> None:
    docs = [Document(page_content="stop_loss = 3.2%", metadata={"src": "a"}), Document(page_content="")]
    out = ObfusAIDocumentTransformer(ObfusAIComponent("low", seed="s")).transform_documents(docs)
    assert "stop_loss" not in out[0].page_content.split("<obfusai-context")[0]
    assert out[0].metadata["src"] == "a" and "obfusai_session_id" in out[0].metadata
    assert out[1].page_content == ""
    bare = ObfusAIDocumentTransformer(ObfusAIComponent("low", seed="s"), include_anchor=False)
    doc = bare.transform_documents(docs[:1])[0]
    assert "<obfusai-context" not in doc.page_content and "<obfusai-context" in doc.metadata["obfusai_anchor"]


def test_runnable_in_chain() -> None:
    prompt = ChatPromptTemplate.from_messages([("system", "Alert when edge_bps > 35. Hi {name}."), ("human", "{q}")])
    step = ObfusAIRunnable(ObfusAIComponent("medium", seed="s"))
    messages = (prompt | step).invoke({"name": "Ann", "q": "edge_bps is 40"})
    assert isinstance(messages[0], SystemMessage)
    assert "edge_bps" not in messages[0].content.split("<obfusai-context")[0]
    assert "Hi Ann" in messages[0].content
    assert messages[1].content == "edge_bps is 40"
    assert "edge_bps" in step.restore(messages[0].content.split("<obfusai-context")[0])


def test_runnable_inputs() -> None:
    step = ObfusAIRunnable(ObfusAIComponent("low", seed="s"))
    assert isinstance(step.invoke("limit_x = 42.5")[0], SystemMessage)
    msgs = step.invoke([HumanMessage(content="limit_x = 42.5")])
    assert len(msgs) == 1 and msgs[0].content == "limit_x = 42.5"  # human text is not obfuscated
    multi = step.invoke([SystemMessage(content=[{"type": "text", "text": "limit_x = 42.5"}])])
    assert "limit_x" not in multi[0].content[0]["text"]
    assert multi[0].content[-1]["text"].startswith("<obfusai-context")


def test_protected_model_round_trip_and_stream() -> None:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from obfusai.adapters.langchain import ObfusAIProtectedModel

    engine = ObfusAIComponent("low", seed="s")
    prompt = ChatPromptTemplate.from_messages([("system", "Alert when edge_bps > 35."), ("human", "{q}")])
    seen = []

    class Echo(GenericFakeChatModel):
        def _generate(self, messages, *a, **k):  # reply with the first alias found in the system prompt
            seen.append(messages[0].content)
            alias = messages[0].content.split(" > ")[0].split()[-1]
            self.messages = iter([AIMessage(content=f"Watching {alias} closely")])
            return super()._generate(messages, *a, **k)

    model = ObfusAIProtectedModel(Echo(messages=iter([])), engine)
    out = model.invoke(prompt.invoke({"q": "go"}))
    assert out.content == "Watching edge_bps closely"
    assert "edge_bps" not in seen[0].split("<obfusai-context")[0]

    chunks = list(model.stream(prompt.invoke({"q": "go"})))
    assert "".join(c.content for c in chunks) == "Watching edge_bps closely"
