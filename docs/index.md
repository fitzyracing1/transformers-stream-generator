# fitzyracing-transformers-stream-generator

`fitzyracing-transformers-stream-generator` is a drop-in fork of [transformers-stream-generator](https://github.com/LowinLi/transformers-stream-generator) and part of [360 Bench](https://github.com/fitzyracing1/360-bench). It is based on upstream `transformers-stream-generator==0.0.5`, and the API is unchanged. All credit for the original library goes to its authors and contributors.

## What the fork fixes (summary of [README.md](../README.md))

- Imports on transformers 4.57 and 5.x.
- On transformers 4.41+, `generate(do_stream=True)` streams through transformers' own `generate(streamer=...)`, and non-streaming calls are passed through unchanged.
- Runs the original 0.0.5 code on transformers 4.26 to 4.40.
- Beam search with `do_stream=True` raises a clear `ValueError`.

The full details and links to the upstream issues are in [README.md](../README.md).

Pages: [Getting started](getting-started.md) · [Architecture](architecture.md) · [Contributing](contributing.md)

This repo has no CI workflow.

## License

MIT. The original notice is kept: Copyright (c) 2023 lowinli.
Modifications: Copyright (c) 2026 Joshua Almeida.
