from setuptools import find_packages, setup

setup(
    name="graphseg-python",
    version="0.0.1",
    description="graphseg",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: ISC License (ISCL)",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
    ],
    packages=find_packages("graphseg"),
    install_requires=["genutility>=0.0.83", "networkx", "numpy>1.16.0", "scipy"],
)
