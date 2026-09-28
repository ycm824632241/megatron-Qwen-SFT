"""Runtime monkey-patches applied before Megatron / TE import.

Each module exposes ``apply() -> str`` and is idempotent where possible.
See ``env/bootstrap.py`` for call order and why each patch exists.
"""
