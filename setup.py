# installs the package, including its dependencies
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="a2pm",
    version="1.0.0",
    description="Adaptative Perturbation Pattern Method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vitorinojoao/a2pm",
    author="João Vitorino",
    author_email='jpmvo@outlook.com',
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["a2pm", "a2pm.patterns"],
    install_requires=["numpy>=1.22.2", "scikit-learn>=0.24.2"],
)
