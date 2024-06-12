import setuptools

with open("README.md","r",encoding="utf-8") as r:
    long_description=r.read()

setuptools.setup(
    name="monaka",
    version="0.0.3",
    description="A Japanese parser (including support for historical Japanese)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/komiya-lab/monaka",
    license="MIT",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Japanese",
    ],
    package_data={
        "monaka":["./resource/mecabrc"],
    },
    entry_points={
        "console_scripts":[
            "monaka_train = monaka.train_cli:app",
            "monaka = monaka.cli:app"
        ],
    },
    project_urls={
        "Source":"https://github.com/komiya-lab/monaka",
        "Tracker":"https://github.com/komiya-lab/monaka/issues",
    },
    install_requires=[
        "protobuf==4.25.3",
        "transformers==4.35.0",
        "registrable",
        "mecab-python3",
        "fugashi",
        "ipadic",
        "typer",
        "torch",
        "prettytable"
    ],
    extras_require={
        "Train": [
            "tensorboard",
            "tqdm"
        ]
    }
)
