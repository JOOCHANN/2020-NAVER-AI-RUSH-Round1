#nsml: registry.navercorp.com/nsml/airush2020:pytorch1.5
from setuptools import setup

setup(
    name = 'pytorch pjc',
    version = '0.1',
    install_requires=[
        'flask',
        'tqdm',
        'fire',
        'pandas',
        'xlrd',
        'openpyxl',
        'pyhdfs',
        'pymongo',
        'redis',
        'scikit-learn',
        'torch==1.5.0',
        'torchtext',
        'revtok',
        'efficientnet_pytorch'
    ]
)