"""
test_black_box.py
"""


from aiml.surrogate_model.create_surrogate_model import create_surrogate_model
from aiml.attack.test_white_box import test_all_white_box_attack

def black_box_test(black_box_model):
    #TODO: Implement black-box attack testing
    surrogate_model = create_surrogate_model(black_box_model)
    result = test_all_white_box_attack(surrogate_model)