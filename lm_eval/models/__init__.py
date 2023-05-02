from . import gpt2
from . import gpt3
from . import huggingface
from . import textsynth
from . import dummy
from . import lit_llama

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "hf-causal": gpt2.HFLM,
    "hf-causal-experimental": huggingface.AutoCausalLM,
    "hf-seq2seq": huggingface.AutoSeq2SeqLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
    "lit-llama": lit_llama.LitLLaMA
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
