from distutils.core import setup

setup(
    name='ThreadWeave',
    author='Eelco Hoogendoorn',
    author_email='hoogendoorn.eelco@gmail.com',
    packages=['threadweave', 'threadweave.test', 'threadweave.frontend','threadweave.backend'],
    scripts=[],
    url='http://pypi.python.org/pypi/ThreadWeave/',
    license='LICENSE.txt',
    description='nd-array aware GPU kernels',
    long_description=open('README.txt').read(),
    install_requires=[
        "pyparsing",
        "numpy",
        
    ],
)
