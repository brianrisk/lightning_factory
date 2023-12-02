from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='lightning_factory',
    version='0.1.1',
    author='Brian Risk',
    author_email='brian@d.at',
    packages=find_packages(),  # This will automatically find packages in the directory
    url='https://d.at/ligntning_factory/',
    license='Apache License 2.0',  # Apache License as specified in requirements
    description='Quickly, with very little code, create PyTorch Lightning models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # If your README is in markdown
    install_requires=required,  # Use the list we read from the requirements.txt
    python_requires='>=3.6',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',  # Choose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    scripts=[],
)
