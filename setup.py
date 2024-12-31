from setuptools import setup, find_packages

setup(
    name="Content based Movie Recommedation system",
    version="1.0",
    description="A machine learning project for Movie Recommedation",
    author="Maaz uddin",
    packages=find_packages(),
    install_requires=[
        "flask",
        "pandas",
        "numpy",
        "scikit-learn",
        "seaborn",
        "matplotlib"
    ]
)
