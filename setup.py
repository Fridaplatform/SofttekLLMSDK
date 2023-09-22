from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.1"
DESCRIPTION = "A python package that allows you to build LLM-Powered Applications"
LONG_DESCRIPTION = (
    "A python package that allows you to build LLM-Powered Applications"
)
extra_requires = {}

with open("requirements.txt") as f:
    required = f.read().splitlines()




# Setting up
setup(
    name="innovation-stk",
    version=VERSION,
    author="Innovation-stk",
    author_email="diego.anaya@softtek.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=required,
    extra_requires=extra_requires,
    keywords=["python"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.10",
)