"""
standard_white_box_test.py

This module provides functions to create various adversarial attacks using the ART library.
"""


from art.attacks.evasion import (
    AutoProjectedGradientDescent,
    CarliniL0Method,
    CarliniL2Method,
    CarliniLInfMethod,
    DeepFool,
    ElasticNet,
    HopSkipJump,
    NewtonFool,
    PixelAttack,
    SaliencyMapMethod,
    SquareAttack,
    UniversalPerturbation,
    ZooAttack,
)
import numpy as np

def auto_projected_cross_entropy(estimator, norm = np.inf, eps = 0.3, eps_step = 0.1, batch_size = 32):
    return AutoProjectedGradientDescent(
                    estimator=estimator,  # type: ignore
                    norm=norm,
                    eps=eps,
                    eps_step=eps_step,
                    max_iter=100,
                    targeted=False,
                    nb_random_init=5,
                    batch_size=batch_size,
                    loss_type="cross_entropy",
                )

def auto_projected_difference_logits_ratio(estimator, norm = np.inf, eps = 0.3, eps_step = 0.1, batch_size = 32):
    return AutoProjectedGradientDescent(
                    estimator=estimator,  # type: ignore
                    norm=norm,
                    eps=eps,
                    eps_step=eps_step,
                    max_iter=100,
                    targeted=False,
                    nb_random_init=5,
                    batch_size=batch_size,
                    loss_type="cross_entropy",
                )

def deepfool_auto(estimator, batch_size = 32):
    return DeepFool(
                        classifier=estimator,  # type: ignore
                        max_iter=100,
                        epsilon=1e-3,
                        nb_grads=10,
                        batch_size=batch_size,
                    )

def square_attack_auto(estimator, norm = np.inf, eps = 0.3):
    return SquareAttack(estimator=estimator, norm=norm, max_iter=5000, eps=eps, p_init=0.8, nb_restarts=5)

def carlini_L0_attack(classifier, confidence=0.0, targeted=False, learning_rate=0.01, binary_search_steps=10,
                      max_iter=10, initial_const=0.01, mask=None, warm_start=True, max_halving=5,
                      max_doubling=5, batch_size=1, verbose=True):
    return CarliniL0Method(classifier, confidence, targeted, learning_rate, binary_search_steps, max_iter,
                           initial_const, mask, warm_start, max_halving, max_doubling, batch_size, verbose)

def carlini_L2_attack(classifier, confidence=0.0, targeted=False, learning_rate=0.01, binary_search_steps=10,
                      max_iter=10, initial_const=0.01, max_halving=5, max_doubling=5, batch_size=1,
                      verbose=True):
    return CarliniL2Method(classifier, confidence, targeted, learning_rate, binary_search_steps, max_iter,
                           initial_const, max_halving, max_doubling, batch_size, verbose)

def carlini_Linf_attack(classifier, confidence=0.0, targeted=False, learning_rate=0.01, max_iter=10,
                        decrease_factor=0.9, initial_const=1e-05, largest_const=20.0, const_factor=2.0,
                        batch_size=1, verbose=True):
    return CarliniLInfMethod(classifier, confidence, targeted, learning_rate, max_iter, decrease_factor, initial_const,
                             largest_const, const_factor, batch_size, verbose)

def deep_fool_attack(classifier, max_iter=100, epsilon=1e-06, nb_grads=10, batch_size=1, verbose=True):
    """
    Deep_Fool takes in a classifier and returns an ART DeepFool instance.

    Inputs: 
    classifier: A trained classifier of type CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE.
    max_iter (int): The maximum number of iterations.
    epsilon (float): Overshoot parameter.
    nb_grads (int): The number of class gradients (top nb_grads w.r.t. prediction) to compute. This way only the most likely classes are considered, speeding up the computation.
    batch_size (int): Batch size
    verbose (bool): Show progress bars.

    Output:
    DeepFool: A DeepFool instance from art.attacks.evasion
    """
    return DeepFool(classifier, max_iter, epsilon, nb_grads, batch_size, verbose)

def elastic_net_attack(classifier, confidence=0.0, targeted=False, learning_rate=0.01, binary_search_steps=9,
                      max_iter=100, beta=0.001, initial_const=0.001, batch_size=1, decision_rule='EN',
                      verbose=True):
    return ElasticNet(classifier, confidence, targeted, learning_rate, binary_search_steps, max_iter, beta,
                      initial_const, batch_size, decision_rule, verbose)

def hopskipjump_attack(classifier, batch_size=64, targeted=False, norm=2, max_iter=50, max_eval=10000,
                      init_eval=100, init_size=100, verbose=True):
    return HopSkipJump(classifier, batch_size, targeted, norm, max_iter, max_eval, init_eval, init_size, verbose)

def newton_fool_attack(classifier, max_iter=100, eta=0.01, batch_size=1, verbose=True):
    return NewtonFool(classifier, max_iter, eta, batch_size, verbose)

def pixel_attack(classifier, th = None, es = 1, max_iter = 100, targeted = False, verbose = True):
    return PixelAttack(classifier, th, es, max_iter, targeted, verbose)

def saliency_map_attack(classifier, theta=0.1, gamma=1.0, batch_size=1, verbose=True):
    return SaliencyMapMethod(classifier, theta, gamma, batch_size, verbose)

def square_attack(estimator, norm = np.inf, adv_criterion = None, loss = None, max_iter = 100, eps = 0.3, p_init = 0.8, nb_restarts = 1, batch_size = 128, verbose = True):
    return SquareAttack(estimator, norm, adv_criterion, loss, max_iter, eps, p_init, nb_restarts, batch_size, verbose)

def universal_perturbation_attack(classifier, attacker='deepfool', attacker_params=None, delta=0.2, max_iter=20,
                                  eps=10.0, norm=np.inf, batch_size=32, verbose=True):
    return UniversalPerturbation(classifier, attacker, attacker_params, delta, max_iter, eps, norm, batch_size, verbose)

def zoo_attack(classifier, confidence=0.0, targeted=False, learning_rate=0.01, max_iter=10, binary_search_steps=1,
               initial_const=0.001, abort_early=True, use_resize=True, use_importance=True, nb_parallel=128,
               batch_size=1, variable_h=0.0001, verbose=True):
    return ZooAttack(classifier, confidence, targeted, learning_rate, max_iter, binary_search_steps, initial_const,
                      abort_early, use_resize, use_importance, nb_parallel, batch_size, variable_h, verbose)
