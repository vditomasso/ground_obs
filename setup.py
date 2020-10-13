from setuptools import setup

setup(name='ground_obs',
      version='0.1',
      description='A package built to determine optimal ground-based observations of exoplanet atmospheres',
      author='Victoria DiTomasso',
      author_email='victoria.ditomasso@cfa.harvard.edu',
      license='GPLv3',
      packages=['ground_obs'],
      install_requires=['numpy', 'matplotlib', 'astropy', 'pandas', 'seaborn'],
      test_suite='nose.collector',
      tests_require=['nosetests'])