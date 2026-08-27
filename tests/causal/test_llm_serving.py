import sys
import importlib
from agents.causal.llm import (
    LLMClient, NullLLMClient, make_llm_client, VLLMClient, HFClient,
)


def test_llm_module_is_offline_safe():
    # importar o módulo não pode puxar vllm/torch (topo stdlib-only)
    importlib.import_module("agents.causal.llm")
    assert "vllm" not in sys.modules
    assert "torch" not in sys.modules


def test_make_client_none_returns_null():
    assert isinstance(make_llm_client(None), NullLLMClient)
    assert isinstance(make_llm_client(), NullLLMClient)


def test_make_client_missing_model_falls_back_to_null():
    # sem vllm/transformers (ou caminho inexistente) → NullLLMClient, nunca levanta
    c = make_llm_client("/nao/existe/qwen")
    assert isinstance(c, NullLLMClient)


def test_concrete_clients_are_llmclients():
    assert issubclass(VLLMClient, LLMClient)
    assert issubclass(HFClient, LLMClient)
