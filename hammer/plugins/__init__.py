"""Plugin discovery interface for HammerAndNail.

Plugins are Python modules that expose a `register(registry)` function.
Place plugin modules in this package or install them as separate packages
with the entry point group 'hammer.plugins'.

Example plugin:
    # hammer/plugins/my_tool.py
    from hammer.tools.registry import ToolDefinition

    def register(registry):
        registry.register(ToolDefinition(
            name="my_tool",
            description="Does something custom",
            fn=my_tool_fn,
        ))
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path

logger = logging.getLogger("hammer.plugins")


def load_plugins(registry) -> int:
    """Discover and load all plugins in this package.

    Returns the number of plugins loaded.
    """
    loaded = 0
    package_path = Path(__file__).parent

    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        try:
            module = importlib.import_module(f"hammer.plugins.{module_name}")
            if hasattr(module, "register"):
                module.register(registry)
                logger.info("Loaded plugin: %s", module_name)
                loaded += 1
        except Exception:
            logger.exception("Failed to load plugin: %s", module_name)

    return loaded
