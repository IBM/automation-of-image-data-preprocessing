"""
The IBM License 2017.
Contact: Tran Ngoc Minh (M.N.Tran@ibm.com).
"""
from setuptools import setup, find_packages

setup(
    name="sentana",
    version="0.1",
    packages=find_packages(),
    install_requires=["tensorflow>=0.12"],
    author="Tran Ngoc Minh",
    author_email="M.N.Tran@ibm.com",
    description="This is a package for visual sentiment analysis.",
    license="IBM Ireland",
    keywords="visual sentiment analysis",
    url="https://github.ibm.com/M-N-Tran/sentana",
)