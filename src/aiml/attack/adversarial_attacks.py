"""
adversarial_attacks.py

This module provides definition for various adversarial attacks using 
the ART library.
"""


from art.attacks.evasion import (
    PixelAttack,
    AutoProjectedGradientDescent,
    ZooAttack,
    CarliniL0Method,
    CarliniL2Method,
    CarliniLInfMethod,
    SquareAttack,
    DeepFool
)
import numpy as np


def pixel_attack(
    classifier, th=None, es=1, max_iter=100, targeted=False, verbose=True
):
    return PixelAttack(classifier, th, es, max_iter, targeted, verbose)


def auto_projected_cross_entropy(
    estimator, norm=np.inf, eps=0.3, eps_step=0.1, batch_size=32
):
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


def auto_projected_difference_logits_ratio(
    estimator, norm = np.inf, eps = 0.3, eps_step = 0.1, batch_size = 32
):
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


def zoo_attack(
    classifier, confidence=0.0, targeted=False, learning_rate=0.01, max_iter=10, 
    binary_search_steps=1, initial_const=0.001, abort_early=True, use_resize=True, 
    use_importance=True, nb_parallel=128, batch_size=1, variable_h=0.0001, verbose=True
):
    return ZooAttack(
        classifier, confidence, targeted, learning_rate, max_iter, binary_search_steps, 
        initial_const, abort_early, use_resize, use_importance, nb_parallel, batch_size, 
        variable_h, verbose
    )


def carlini_L0_attack(
    classifier, confidence=0.0, targeted=False, learning_rate=0.01, binary_search_steps=10,
    max_iter=10, initial_const=0.01, mask=None, warm_start=True, max_halving=5, max_doubling=5, 
    batch_size=1, verbose=True
):
    return CarliniL0Method(
        classifier, confidence, targeted, learning_rate, binary_search_steps, max_iter, 
        initial_const, mask, warm_start, max_halving, max_doubling, batch_size, verbose
    )


def carlini_L2_attack(
    classifier, confidence=0.0, targeted=False, learning_rate=0.01, binary_search_steps=10,
    max_iter=10, initial_const=0.01, max_halving=5, max_doubling=5, batch_size=1, verbose=True
):
    return CarliniL2Method(
        classifier, confidence, targeted, learning_rate, binary_search_steps, max_iter, 
        initial_const, max_halving, max_doubling, batch_size, verbose
    )


def carlini_Linf_attack(
    classifier, confidence=0.0, targeted=False, learning_rate=0.01, max_iter=10, 
    decrease_factor=0.9, initial_const=1e-05, largest_const=20.0, const_factor=2.0, batch_size=1, 
    verbose=True
):
    return CarliniLInfMethod(
        classifier, confidence, targeted, learning_rate, max_iter, decrease_factor, initial_const,
        largest_const, const_factor, batch_size, verbose
    )


def square_attack(
    estimator, norm = np.inf, adv_criterion = None, loss = None, max_iter = 100, eps = 0.3, 
    p_init = 0.8, nb_restarts = 1, batch_size = 128, verbose = True
):
    return SquareAttack(
        estimator, norm, adv_criterion, loss, max_iter, eps, p_init, nb_restarts, batch_size, 
        verbose
    )


def deep_fool(
    classifier, max_iter = 100, epsilon = 1e-06, nb_grads = 10, batch_size = 1, verbose = True
):
    return DeepFool(classifier, max_iter, epsilon, nb_grads, batch_size, verbose)
