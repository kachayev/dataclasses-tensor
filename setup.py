from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as f:
    readme = f.read()

setup(
    name="dataclasses-tensor",
    version="0.2.0",
    packages=find_packages(exclude=("tests*",)),
    package_data={"dataclasses_tensor": ["py.typed"]},
    author="Oleksii Kachaiev",
    author_email="kachayev@gmail.com",
    description="Easily serialize dataclasses to and from tensors",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/kachayev/dataclasses-tensor",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    license="MIT",
    install_requires=[
        'dataclasses;python_version=="3.6"',
        "typing-inspect>=0.4.0",
    ],
    python_requires=">=3.6",
    extras_require={
        "dev": [
            "pytest",
            "ipython",
            "flake8",
            "numpy==1.18.5",
            "torch==1.8.0",
        ]
    },
    include_package_data=True
)
