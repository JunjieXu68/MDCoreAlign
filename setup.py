"""Setuptools hooks for the stable MDCoreAlign wheel surface."""

from __future__ import annotations

from setuptools import setup
from setuptools.command.build_py import build_py

EXCLUDED_RESEARCH_MODULES = {
    "mdcorealign.benchmark_minimal",
    "mdcorealign.experimental_gpu_batch",
    "mdcorealign.experimental_stream",
    "mdcorealign.experimental_throughput_v2",
    "mdcorealign.guarded",
}


class StableBuildPy(build_py):
    """Exclude isolated research modules from distributable wheels."""

    def find_package_modules(self, package, package_dir):
        # Both build_packages (the actual copy) and find_all_modules use this
        # hook. Filtering only find_all_modules does not filter wheel contents.
        modules = super().find_package_modules(package, package_dir)
        return [
            item
            for item in modules
            if f"{item[0]}.{item[1]}" not in EXCLUDED_RESEARCH_MODULES
        ]


setup(cmdclass={"build_py": StableBuildPy})
