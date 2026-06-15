from setuptools import setup, find_packages

setup(
    name="nesycr",
    version="0.1.0",
    description="Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning",
    author="Jooyoung Kim",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.11",
    install_requires=[],
)
