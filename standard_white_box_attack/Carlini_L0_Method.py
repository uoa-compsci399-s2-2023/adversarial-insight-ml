from art.attacks.evasion import *
from art.defences.trainer import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE
import numpy as np

def Carlini_L0_Method(classifier, confidence = 0.0, targeted = False, learning_rate = 0.01, binary_search_steps = 10, max_iter = 10, initial_const: float = 0.01, mask = None, warm_start = True, max_halving = 5, max_doubling = 5, batch_size = 1, verbose = True):
    return CarliniL0Method(classifier,confidence,targeted,learning_rate,binary_search_steps,max_iter, initial_const, mask, warm_start, max_halving, max_doubling, batch_size, verbose)