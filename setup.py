from setuptools import setup, find_packages

setup(
    name="autodp",
    version="0.1",
    packages=find_packages(),
    install_requires=["tensorflow>=0.12"],
    author="Tran Ngoc Minh",
    author_email="M.N.Tran@ibm.com",
    description="This project focuses on the automation of data preprocessing.",
    license="IBM Research Ireland",
    keywords="Automation Image Data Preprocessing",
)
