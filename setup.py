# Install the package, including its dependencies
from setuptools import setup

with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="a2pm",
    version="1.2.0",
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
        "License :: OSI Approved",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Adaptive Technologies",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=["a2pm", "a2pm.callbacks", "a2pm.patterns", "a2pm.wrappers"],
    python_requires=">=3.5",
    install_requires=["numpy>=1.17.5,<2", "scikit-learn>=0.23.2,<2"],
)
