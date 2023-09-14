"""
Adversarial Insight ML

This is a Python package that provides evaluation of ML models based on adversarial attacks.

This module is the entry point for the package.

Author: Team 7

"""


# Import all individual modules (.py) within pacakge
from .attack import *
from .evaluate import *
from .load_data import *
from .standard_white_box_attack import *
from .surrogate_model import *
from .test_accuracy import *

# Define package-wide variables
__version__ = "0.0.1"
__doc__ = "Adversarial Insight ML is a package that provides evaluation of ML models based on adversarial attacks."

# List of symbols to be imported when using "from package import *"
__all__ = ['attack', 'evaluate', 'load_data', 'standard_white_box_attack', 'surrogate_model', 'test_accuracy']  

# Initialization setup (package-wide configuration)
def setup_package():
    """
    Setup the package.

    This function can be used to perform any necessary initialization or setup
    when the package is imported.
    """
    print("AIML package ({})is being initialized.".format(__version__))

setup_package()
