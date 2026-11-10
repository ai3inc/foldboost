"""Opt-in exact-module hook. No eager Torch/Triton import, no disk source edits."""
import importlib.machinery
import os
INSTALLED = False
if os.environ.get('FLASHP_MPNN036') == '1':
    original = importlib.machinery.SourceFileLoader.exec_module
    def execute(loader, module):
        result = original(loader, module)
        if module.__name__ == 'triton.compiler.compiler':
            from cached_key import install
            install(module)
            global INSTALLED
            INSTALLED = True
        return result
    importlib.machinery.SourceFileLoader.exec_module = execute
