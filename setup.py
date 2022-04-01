# Install the package, including its dependencies
from setuptools import setup

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="a2pm",
    version="1.1.1",
    description="Adaptative Perturbation Pattern Method",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="João Vitorino",
    author_email="jpmvo@outlook.com",
    license="MIT",
    url="https://github.com/vitorinojoao/a2pm",
    project_urls={
        "Article": "https://doi.org/10.3390/fi14040108",
        "Documentation": "https://a2pm.readthedocs.io/en/latest/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["a2pm", "a2pm.callbacks", "a2pm.patterns"],
    install_requires=["numpy>=1.17.5<2", "scikit-learn>=0.23.2<2"],
)
