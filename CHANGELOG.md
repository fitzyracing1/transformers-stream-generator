# Changelog

## 0.0.6 (2026-10-09, fitzyracing-transformers-stream-generator fork)

Based on transformers-stream-generator 0.0.5. Distribution name
`fitzyracing-transformers-stream-generator`; the import name is still
`transformers_stream_generator`.

- `import transformers_stream_generator` works again on transformers 4.57+ and 5.x
  (`ImportError: cannot import name 'BeamSearchScorer'` / `'DisjunctiveConstraint'`, upstream #15).
- `model.generate(..., do_stream=True)` (and Qwen-1's `chat_stream()` usage) works on
  transformers 4.41 to 5.x: tokens are streamed through transformers' own public
  `generate(streamer=...)`. It yields the same tokens as non-streaming sampling with the same
  seed. Closing the generator early stops generation.
- After `init_stream_support()`, ordinary `model.generate()` calls (greedy, sampling, beam
  search, ...) work again on transformers 4.41+, where 0.0.5 broke every call; they are passed to
  transformers' `generate()`.
- `do_stream=True` with `num_beams > 1` now raises a clear `ValueError` on transformers 4.41+
  (0.0.5 silently returned `None`). `model.sample_stream()` raises `NotImplementedError` there
  (use `generate(..., do_stream=True)`).
- transformers < 4.41: unchanged; the 0.0.5 code runs as before.
- Packaging: `pyproject.toml` replaces `setup.py`; license metadata now says MIT only (0.0.5's
  classifiers also listed Apache, contradicting its MIT LICENSE). The upstream publish workflow is
  removed.
- Tests: add `tests/test_fork_fixes.py` (tiny public test model, about 2 MB).

## 0.0.5 (2024-03-11) and earlier

See the [upstream history](https://github.com/LowinLi/transformers-stream-generator/commits/main).
