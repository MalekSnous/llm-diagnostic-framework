"""
Setup script for llm-diagnostic package.
For most installations, use: pip install -e .
"""

from setuptools import setup

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="llm-diagnostic",
    version="0.1.0",
    author="Malek Senoussi",
    author_email="malek.senoussi@gmail.com",
    description="Systematic framework for diagnosing and improving LLM performance",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MalekSnous/llm-diagnostic-framework",
    packages=[
        "llm_diagnostic",
        "llm_diagnostic.core",
        "llm_diagnostic.failure_tests",
        "llm_diagnostic.improvements",
        "llm_diagnostic.visualization",
        "llm_diagnostic.models",
    ],
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "llm-diagnose=scripts.run_diagnostics:main",
            "llm-improve=scripts.run_improvements:main",
            "llm-report=scripts.generate_report:main",
        ],
    },
)