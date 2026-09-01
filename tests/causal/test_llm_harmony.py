from agents.causal.llm import (
    _should_use_harmony,
    extract_final_channel,
    parse_goal,
)


# --- extract_final_channel: isola o canal `final` do formato Harmony (gpt-oss) ---
def test_extract_final_channel_drops_analysis_cot():
    # saida Harmony decodificada COM marcadores: CoT no canal analysis, resposta no final
    raw = (
        "<|channel|>analysis<|message|>Vou clicar no centro do objeto raro."
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        '{"type":"press","action":"ACTION1"}<|return|>'
    )
    assert extract_final_channel(raw) == '{"type":"press","action":"ACTION1"}'


def test_extract_final_channel_terminated_by_end():
    raw = "<|channel|>final<|message|>ola mundo<|end|>"
    assert extract_final_channel(raw) == "ola mundo"


def test_extract_final_channel_passthrough_when_no_markers():
    # modelo nao-harmony (Qwen): texto cru sem canais -> devolve como veio
    assert extract_final_channel('{"type":"press","action":"ACTION2"}') == (
        '{"type":"press","action":"ACTION2"}'
    )


# --- _should_use_harmony: gpt-oss usa harmony; Qwen/None nao ---
def test_should_use_harmony_true_for_gpt_oss():
    assert _should_use_harmony("/kaggle/input/gpt-oss-120b/transformers/v1") is True


def test_should_use_harmony_false_for_qwen_and_none():
    assert _should_use_harmony("/kaggle/input/qwen-3/transformers/32b/1") is False
    assert _should_use_harmony(None) is False


# --- cadeia end-to-end: decode Harmony -> extract final -> parse_goal vira acao valida ---
def test_harmony_output_parses_to_goal():
    raw = (
        "<|channel|>analysis<|message|>pensando...<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        '{"type":"press","action":"ACTION3"}<|return|>'
    )
    goal = parse_goal(extract_final_channel(raw))
    assert goal == {"type": "press", "action": "ACTION3"}
