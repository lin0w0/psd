"""Test-time defense implementations for palmprint recognition.

Planned interface
-----------------
Defenses wrap a loaded CompNet and either transform the input tensor before
``getFeatureCode`` or post-process the feature code before matching. The
recognition protocols and metrics in :mod:`evaluation` can be reused
unchanged to measure clean/robust performance.

Useful imports::

    from model import load_compnet
    from evaluation import extract_features, run_protocols
"""
