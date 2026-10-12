# fitzyracing-transformers-stream-generator

> Part of **[360 Bench](https://github.com/fitzyracing1/360-bench)**, tested fixes for abandoned PyPI packages.

[![PyPI](https://img.shields.io/pypi/v/fitzyracing-transformers-stream-generator.svg)](https://pypi.org/project/fitzyracing-transformers-stream-generator/)

> **This is a fork of [transformers-stream-generator](https://github.com/LowinLi/transformers-stream-generator)
> by [LowinLi](https://github.com/LowinLi)**, published as a drop-in replacement that works with
> current transformers. The upstream package (~360k downloads a month, required by the remote
> code of Qwen(-1) models, among others) has not been released since 0.0.5 (March 2024), and on
> transformers 4.57 and newer it can't even be imported. The import name is still
> `transformers_stream_generator`, so no code changes are needed. All credit for the original
> library goes to its author; it remains available under the same MIT license.
>
> If a fixed `transformers-stream-generator` release appears on PyPI, prefer it and switch back.


## What's fixed in this fork

Based on upstream `transformers-stream-generator==0.0.5` (tag
[`0.0.5`](https://github.com/LowinLi/transformers-stream-generator/tree/0.0.5)). The API is unchanged.

0.0.5 copies the `generate()` method of transformers 4.26 and builds on transformers internals
that have since changed. What that meant, tested with a tiny GPT-2:

| transformers | 0.0.5 | this fork |
|---|---|---|
| 4.26 to 4.40 | works | **unchanged**: runs exactly the same 0.0.5 code |
| 4.41 to 4.56 | imports, but after `init_stream_support()` **every** `model.generate()` call fails (`TypeError`/`AttributeError`), streaming or not | works |
| 4.57 and 5.x | `ImportError: cannot import name 'BeamSearchScorer'` (4.57) / `'DisjunctiveConstraint'` (5.x) on import | works |

On transformers 4.41 and newer the fork:

- **imports** without the classes transformers removed (they were only used by the old code);
- **streams** `model.generate(..., do_stream=True)` through transformers' own public
  `generate(streamer=...)`, running generation in a background thread and yielding each step's
  token ids (a tensor of shape `(batch_size,)` on the input's device), exactly like 0.0.5. With
  the same `seed` it yields the same tokens as non-streaming sampling. Qwen(-1)'s `chat_stream()`,
  which calls `NewGenerationMixin.generate` with a `StreamGenerationConfig(do_stream=True)`, works
  the same way. If you stop reading early (`break`, `generator.close()`), generation stops too;
  errors are raised in your loop;
- **passes every other `generate()` call** (greedy, sampling, beam search, ...) to transformers'
  own `generate()`, so `init_stream_support()` no longer breaks them;
- keeps 0.0.5's conventions: stream mode always samples (as 0.0.5 did, even with
  `do_sample=False`; see [#1](https://github.com/LowinLi/transformers-stream-generator/issues/1)),
  and `generate()` seeds the random generators with `seed=0` unless you pass another `seed`
  (`seed=-1` leaves them alone).

What doesn't work on transformers 4.41+, with a clear error instead of a crash:

- **Beam search can't stream.** `do_stream=True` with `num_beams > 1` raises
  `ValueError: do_stream=True only supports sampling with num_beams=1 ...` (0.0.5 silently
  returned `None`, see [#3](https://github.com/LowinLi/transformers-stream-generator/issues/3)).
  Beam search without `do_stream` works.
- **`model.sample_stream()`** (the internal sampler, which 0.0.5 also exposed) relies on
  transformers internals that no longer exist, so calling it raises `NotImplementedError` telling
  you to use `generate(..., do_stream=True)`. Assigning it, as Qwen(-1)'s remote code does, still
  works.

As a side effect, the new stream also handles a list of several `eos_token_id`s correctly
([#10](https://github.com/LowinLi/transformers-stream-generator/issues/10)) and batches of more
than one prompt. On transformers < 4.41 these 0.0.5 limitations are kept as they were, because
that code is deliberately left untouched.

Upstream issues: [#15](https://github.com/LowinLi/transformers-stream-generator/issues/15) (no fix
PR existed). Note that Qwen(-1)'s own remote code may have other incompatibilities with recent
transformers; this fork only makes `transformers_stream_generator` itself work.

**Packaging:** `pyproject.toml` replaces `setup.py`; dependencies are unchanged
(`transformers>=4.26.1`). The license metadata now says MIT only, matching the LICENSE file
(0.0.5's classifiers also listed Apache).


## Install

```bash
pip install fitzyracing-transformers-stream-generator
```

### Switching from `transformers-stream-generator`

This distribution installs the same `transformers_stream_generator` package as the original, so
your code stays the same. **Uninstall the original first**, then install the fork:

```bash
pip uninstall -y transformers-stream-generator
pip install fitzyracing-transformers-stream-generator
```

The order matters. Both distributions own the same files, so if you install the fork first and
uninstall the original afterwards, pip deletes the shared files and the import breaks. If that
happens, run `pip install --force-reinstall --no-deps fitzyracing-transformers-stream-generator`.

In `requirements.txt` / `pyproject.toml`, replace `transformers-stream-generator` (or
`transformers_stream_generator`) with `fitzyracing-transformers-stream-generator`.

### If you get it through another package

pip cannot replace a dependency with a differently named package. If a dependency requires
`transformers-stream-generator`, install the fork alongside it and then remove the original's
files:

```bash
pip install fitzyracing-transformers-stream-generator
pip uninstall -y transformers-stream-generator
pip install --force-reinstall --no-deps fitzyracing-transformers-stream-generator   # restore the files the uninstall removed
```

Afterwards `pip check` reports `<package> requires transformers-stream-generator, which is not
installed`; that is expected. Repeat the steps if a later install pulls the original back in.

**uv users** can do this properly with an override that drops the original:

```toml
# pyproject.toml
[project]
dependencies = ["fitzyracing-transformers-stream-generator", "...the package that depends on it..."]

[tool.uv]
override-dependencies = ["transformers-stream-generator; sys_platform == 'never'"]
```

(or `uv pip install --override overrides.txt ...` with that same line in `overrides.txt`).

### Tests

```bash
pip install -e ".[test]"
pytest
```

`tests/test_fork_fixes.py` uses `hf-internal-testing/tiny-random-gpt2` (about 2 MB). It passes
on transformers 4.26, 4.32, 4.36, 4.38, 4.40, 4.41, 4.45, 4.52, 4.57 and 5.19 with this fork;
with 0.0.5 it fails on everything from 4.41 on.


## transformers-stream-generator

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/transformers-stream-generator.svg)](https://pypi.org/project/transformers-stream-generator/)
[![PyPI](https://img.shields.io/pypi/v/transformers-stream-generator.svg)](https://pypi.org/project/transformers-stream-generator/)
[![GitHub license badge](https://img.shields.io/github/license/LowinLi/transformers-stream-generator)](https://github.com/LowinLi/transformers-stream-generator/blob/main/LICENSE)
[![Blog](https://img.shields.io/badge/blog-LowinLi-important)](https://lowin.li)

### Description
This is a text generation method which returns a generator, streaming out each token in real-time during inference, based on Huggingface/Transformers. 

### Web Demo
+ original
![](./pic/original.gif)
+ stream
![](./pic/stream.gif)

### Installation
```bash
pip install transformers-stream-generator
```

### Usage
1. just add two lines of code before your original code
```python
from transformers_stream_generator import init_stream_support
init_stream_support()
```

2. add `do_stream=True` in `model.generate` function and keep `do_sample=True`, then you can get a generator
```python
generator = model.generate(input_ids, do_stream=True, do_sample=True)
for token in generator:
    word = tokenizer.decode(token)
    print(word)
```

### Example
+ run python script [example](./example/run.py) by gpt2
+ run web [example](./example/run_web.py)  by gpt2 and test in client [example](./example/test_client.py)

## Documentation

- In-repo docs: [docs/](docs/index.md)
- Wiki: [https://github.com/fitzyracing1/transformers-stream-generator/wiki](https://github.com/fitzyracing1/transformers-stream-generator/wiki)
