from setuptools import setup, find_packages

with open("README.md") as readme_file:
    README = readme_file.read()

setup_args = dict(
    name="processtransformer",
    version="0.1.3",
    description="Process Transformer Network for Predictive Business Process Monitoring Tasks",
    long_description_content_type="text/markdown",
    long_description=README,
    license="Apache-2.0",
    packages=find_packages(),
    author="Zaharah Bukhsh",
    author_email="z.bukhsh@tue.nl",
    keywords=["Business Process Mointoring", "Predictive Business Process", "Transformer", 
        "Attention-Mechanism", "Neural Network", "Process Transformer"],
    url="https://github.com/Zaharah/processtransformer",
    download_url="https://pypi.org/project/processtransformer/"
)

install_requires = [
    "tensorflow>=2.4",
    "numpy",
    "scikit-learn",
    "pandas"
]

if __name__ == "__main__":
    setup(**setup_args, install_requires=install_requires)