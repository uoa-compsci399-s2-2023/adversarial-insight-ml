"""
__init__.py

Adversarial Insight ML
This is a Python package that provides evaluation of ML models based on 
adversarial attacks. This module is the entry point for the package.
"""


# Import all individual modules (.py) within pacakge
from aiml import attack
from aiml import evaluation
from aiml import load_data
from aiml import surrogate_model

# Define package-wide variables
__version__ = "0.2.0"
__doc__ = "Adversarial Insight ML is a package that provides evaluation of ML models based on adversarial attacks."

# List of symbols to be imported when using "from package import *"
__all__ = [
    "attack",
    "evaluation",
    "load_data",
    "surrogate_model",
]


# Initialization setup (package-wide configuration)
def setup_package():
    """
    Setup the package.

    This function can be used to perform any necessary initialization or setup
    when the package is imported.
    """
    print("AIML package ({}) is being initialized.".format(__version__))


setup_package()
