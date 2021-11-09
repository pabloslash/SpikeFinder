from setuptools import setup, find_packages

# install_requires = open_requirements('requirements.txt')

# d = {}
# exec(open("spikefinder/version.py").read(), None, d)
# version = d['version']

long_description = open("README.md").read()

setup(name='SpikeFinder',
      version='0.1',
      description='Python toolkit for analysis, visualization and extraction of neural features from continuous neural recordings.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/pabloslash/SpikeFinder",
      packages=find_packages(),
      author='Pablo M. Tostado',
      author_email='tostadomarcos@gmail.com',
      license='MIT',
#       install_requires = install_requires,
#       packages=['SpikeFinder'],
      install_requires=['numpy',
                        'matplotlib',
                        'pandas>=0.23',
                        'librosa'
                       ],
      
      classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    )
      zip_safe=False)
