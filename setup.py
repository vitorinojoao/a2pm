# Install the package, including its dependencies
from setuptools import setup

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="a2pm",
    version="1.1.0",
    description="Adaptative Perturbation Pattern Method",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="João Vitorino",
    author_email="jpmvo@outlook.com",
    license="MIT",
    url="https://github.com/vitorinojoao/a2pm",
    project_urls={
        "Article": "https://arxiv.org/abs/2203.04234",
        "Documentation": "https://a2pm.readthedocs.io/en/latest/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["a2pm", "a2pm.callbacks", "a2pm.patterns"],
    install_requires=["numpy>=1.22<2", "scikit-learn>=0.24<2"],
)
