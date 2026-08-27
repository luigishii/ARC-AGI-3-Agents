from agents.causal.llm import LLMClient, NullLLMClient


class _Counter(LLMClient):
    def __init__(self):
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return f"r{self.calls}"


def test_complete_many_default_calls_n_times():
    c = _Counter()
    out = c.complete_many("p", 3)
    assert out == ["r1", "r2", "r3"]
    assert c.calls == 3


def test_null_complete_many():
    out = NullLLMClient().complete_many("p", 4)
    assert out == ["", "", "", ""]
