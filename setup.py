from setuptools import setup, find_packages
import sys

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_requirements = []

requirements = [
    "gt4py",
    "numpy",
    "xarray",
]

test_requirements = []

setup(
    author="Vulcan Technologies LLC",
    author_email="elynnw@vulcan.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="physics standalone is a gt4py-based physical parameterization for atmospheric models",
    install_requires=requirements,
    extras_require={},
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords="physics",
    name="physics",
    packages=find_packages(include=["physics.*"]),
    setup_requires=[],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/VulcanClimateModeling/physics_standalone",
    version="0.1.0",
    zip_safe=False,
)
