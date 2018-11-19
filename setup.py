from setuptools import setup

setup(
    name='bpemb',
    version='0.2.5',
    description='Byte-pair embeddings in 275 languages',
    url='http://github.com/bheinzerling/bpemb',
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
    zip_safe=True)
