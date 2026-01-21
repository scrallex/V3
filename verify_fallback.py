#!/usr/bin/env python3
"""
Test script to verify sep_text_manifold.encode.encode_window
fallback behavior when native bindings are missing.
"""
import sys
import os

# Ensure we can import from src
sys.path.insert(0, os.path.abspath("src"))

try:
    from sep_text_manifold.encode import encode_window
    from sep_text_manifold import native
    print(f"Native module loaded: {native.HAVE_NATIVE}")
    print(f"Use Native enabled: {native.use_native()}")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Dummy Price String (Mocking 60 candles)
# A simple rising then falling sequence
prices = [1.0500 + (i * 0.0001) for i in range(30)] + [1.0530 - (i * 0.0001) for i in range(30)]
price_string = " ".join(f"{p:.5f}" for p in prices)
encoded_payload = price_string.encode("utf-8")

print(f"Payload size: {len(encoded_payload)} bytes")

# Encode
try:
    metrics = encode_window(encoded_payload)
    print("\nMetrics Output:")
    print(metrics)
    
    if metrics["stability"] == 0.0 and metrics["entropy"] == 0.0:
        print("\nFAILURE: Metrics are all zero!")
        sys.exit(1)
    else:
        print("\nSUCCESS: Metrics are non-zero.")
except Exception as e:
    print(f"\nExecution Error: {e}")
    sys.exit(1)
