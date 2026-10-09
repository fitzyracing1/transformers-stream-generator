"""Regression tests for the fitzyracing-transformers-stream-generator fork (360 Bench).

They use the tiny public test model hf-internal-testing/tiny-random-gpt2 (about 2 MB).

On transformers 4.57+ the original 0.0.5 can't even be imported (upstream #15), and on
4.41 to 4.56 every `model.generate()` call fails once `init_stream_support()` has been called.
"""
import threading

import pytest
import torch
from packaging import version
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationMixin, LogitsProcessorList

MODEL = "hf-internal-testing/tiny-random-gpt2"
NEW_TRANSFORMERS = version.parse(transformers.__version__).release >= (4, 41)


def test_import():
    import transformers_stream_generator
    from transformers_stream_generator import init_stream_support  # noqa: F401
    from transformers_stream_generator.main import (  # noqa: F401
        NewGenerationMixin,
        StreamGenerationConfig,
    )

    assert callable(transformers_stream_generator.init_stream_support)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def model():
    from transformers_stream_generator import init_stream_support

    init_stream_support()
    m = AutoModelForCausalLM.from_pretrained(MODEL).eval()
    # Make sure generation never stops early on EOS, so lengths are predictable.
    m.generation_config.eos_token_id = None
    m.generation_config.pad_token_id = 0
    return m


@pytest.fixture(scope="module")
def ids(tok):
    return tok("hello world", return_tensors="pt").input_ids


def _new_tokens(output, ids):
    return output[0, ids.shape[1]:].tolist()


def test_stream_yields_one_token_per_step(model, ids):
    tokens = list(model.generate(ids, max_new_tokens=8, do_sample=True, do_stream=True, seed=0))
    assert len(tokens) == 8
    for t in tokens:
        assert isinstance(t, torch.Tensor)
        assert t.shape == (1,)
        assert t.device == ids.device


def test_stream_matches_non_stream_sampling_with_same_seed(model, ids):
    streamed = [t.item() for t in model.generate(ids, max_new_tokens=8, do_sample=True, top_k=30,
                                                 do_stream=True, seed=0)]
    full = model.generate(ids, max_new_tokens=8, do_sample=True, top_k=30, seed=0)
    assert streamed == _new_tokens(full, ids)


def test_stream_always_samples_like_0_0_5(model, ids):
    # 0.0.5 sampled in stream mode even with do_sample=False; the fork keeps that.
    a = [t.item() for t in model.generate(ids, max_new_tokens=8, do_sample=False, do_stream=True, seed=0)]
    b = [t.item() for t in model.generate(ids, max_new_tokens=8, do_sample=True, do_stream=True, seed=0)]
    assert a == b


def test_qwen_style_generate_stream(model, ids):
    # How Qwen(-1) chat_stream() uses this package.
    from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig

    model.__class__.generate_stream = NewGenerationMixin.generate
    model.__class__.sample_stream = NewGenerationMixin.sample_stream
    config = model.generation_config.to_dict()
    config.pop("do_stream", None)
    config.update(do_sample=True, max_new_tokens=6)
    stream_config = StreamGenerationConfig(**config, do_stream=True)
    seen = []

    class Recorder(transformers.LogitsProcessor):
        def __call__(self, input_ids, scores):
            seen.append(input_ids.shape[1])
            return scores

    torch.manual_seed(0)
    out = []
    for token in model.generate_stream(ids, return_dict_in_generate=False, generation_config=stream_config,
                                       logits_processor=LogitsProcessorList([Recorder()]), seed=-1):
        out.append(token.item())
    assert len(out) == 6
    assert seen[0] == ids.shape[1]  # the custom logits processor was used


@pytest.mark.parametrize("kwargs", [
    dict(do_sample=False),
    dict(do_sample=True),
    dict(num_beams=2, do_sample=False),
])
def test_regular_generate_still_works_after_init(model, ids, kwargs):
    out = model.generate(ids, max_new_tokens=6, seed=0, **kwargs)
    torch.manual_seed(0)
    expected = GenerationMixin.generate(model, ids, max_new_tokens=6, **kwargs)
    assert out.tolist() == expected.tolist()


@pytest.mark.skipif(not NEW_TRANSFORMERS, reason="0.0.5 behaviour is kept unchanged on transformers < 4.41")
def test_stream_with_beams_raises_clear_error(model, ids):
    with pytest.raises(ValueError, match="num_beams"):
        model.generate(ids, max_new_tokens=4, num_beams=2, do_stream=True)


@pytest.mark.skipif(not NEW_TRANSFORMERS, reason="0.0.5 behaviour is kept unchanged on transformers < 4.41")
def test_stream_batch(model, ids):
    batch = torch.cat([ids, ids])
    tokens = list(model.generate(batch, attention_mask=torch.ones_like(batch), max_new_tokens=3,
                                 do_sample=True, do_stream=True, seed=0))
    assert len(tokens) == 3
    assert all(t.shape == (2,) for t in tokens)


@pytest.mark.skipif(not NEW_TRANSFORMERS, reason="0.0.5 behaviour is kept unchanged on transformers < 4.41")
def test_closing_the_stream_stops_generation(model, ids):
    before = threading.active_count()
    gen = model.generate(ids, max_new_tokens=500, do_sample=True, do_stream=True, seed=0)
    first = [next(gen), next(gen)]
    gen.close()
    assert len(first) == 2
    assert threading.active_count() == before


@pytest.mark.skipif(not NEW_TRANSFORMERS, reason="0.0.5 behaviour is kept unchanged on transformers < 4.41")
def test_errors_are_raised_in_the_caller(model, ids):
    gen = model.generate(ids, max_new_tokens=4, do_sample=True, do_stream=True, seed=0,
                         not_a_real_model_kwarg=1)
    with pytest.raises(ValueError):
        list(gen)
