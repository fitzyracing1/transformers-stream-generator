# Getting started

## Install

```
pip uninstall -y transformers-stream-generator && pip install fitzyracing-transformers-stream-generator
```

The import name is still `transformers_stream_generator`. It requires `transformers>=4.26.1`.

## Usage

`from transformers_stream_generator import init_stream_support; init_stream_support()`, then `model.generate(..., do_stream=True)` yields token ids per step. See `example/` and the README.

## Develop and test

```
pip install -e . pytest
pytest
```
