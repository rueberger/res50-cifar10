""" Setup script
"""

from setuptools import setup

setup(name='reslab',
      version='0.1',
      author='Andrew Berger',
      author_email='andbberger@gmail.com',
      url='https://github.com/rueberger/res50-cifar10',
      packages=['reslab'],
      install_requires=[
          'tensorflow',
          'sacred',
          'matplotlib'
          'pymongo'
      ])
