import os
import setuptools

setuptools.setup(
    name='kklib',
    version='0.0.1',
    packages=setuptools.find_packages(),
    author='Kyle Kastner',
    author_email='kastnerkyle@gmail.com',
    description='',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.rst')).read(),
    license='BSD 3-clause',
    url='http://github.com/kastnerkyle/kklib/',
    #package_data={
    #    'pthbldr': ['datasets/data/*']
    #},
    install_requires=['numpy',
                      'scipy',
                      'torch',
                      'unidecode',],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
)
