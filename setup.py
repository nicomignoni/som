import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kohonen-som", 
    version="0.0.1",
    author="Nicola Mignoni",
    author_email="nicola.mignoni@gmail.com",
    description="PyTorch implementation of Kohonen's Self-Organizing Map.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicomignoni/SOM",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
