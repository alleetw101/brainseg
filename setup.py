import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brainseg",
    version="0.0.5",
    author="Alan Lee",
    author_email="alleetw101@gmail.com",
    description="A brain segmentation tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alleetw101/brainseg",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'Tensorflow>=2.0',
        'Pillow>=8.0',
        'niftiprocessing'
    ],
    entry_points={
        'console_scripts': ['brainseg=brainseg:main'],
    },
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
)
