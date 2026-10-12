# Contributing

This fork only takes fixes for confirmed upstream bugs, and each fix needs a test that fails on the original (see the 360 Bench rules). Behaviour should stay the same as upstream wherever upstream did not crash. Before you open a PR, run:

```
pip install -e . pytest
pytest
```

If upstream ships the fix itself, the fork will point people back to upstream.
