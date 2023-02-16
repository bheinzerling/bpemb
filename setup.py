from setuptools import setup

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='bpemb',
    version='0.3.4',
    description='Byte-pair embeddings in 275 languages',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://nlp.h-its.org/bpemb',
    project_urls = {
        "Source Code": "https://github.com/bheinzerling/bpemb",
        "Bug Tracker": "https://github.com/bheinzerling/bpemb",
    },
    author='Benjamin Heinzerling',
    author_email='benjamin.heinzerling@h-its.org',
    license='MIT',
    packages=['bpemb'],
    install_requires=[
        "gensim",
        "numpy",
        "requests",
        "sentencepiece",
        "tqdm"],
    zip_safe=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3"
    ])
