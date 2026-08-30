#!/usr/bin/env python3
import importlib.util
import sys


packages = ["torch", "accelerate", "diffusers", "transformers", "safetensors"]
for package in packages:
    if importlib.util.find_spec(package) is None:
        raise SystemExit(f"ERROR: Python package is not installed: {package}")

print(f"RUNTIME PACKAGES OK python={sys.version.split()[0]} packages={','.join(packages)}")
