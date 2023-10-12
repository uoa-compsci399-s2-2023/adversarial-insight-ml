"""
adversarial_attacks.py

This module contains eight adversarial attacks from the ART library:
    1.AutoProjectedGradientDescent,
    2.CarliniL0Method,
    3.CarliniL2Method,
    4.CarliniLInfMethod,
    5.DeepFool,
    6.PixelAttack,
    7.SquareAttack,
    8.ZooAttack
"""


from art.attacks.evasion import (
    AutoProjectedGradientDescent,
    CarliniL0Method,
    CarliniL2Method,
    CarliniLInfMethod,
    DeepFool,
    PixelAttack,
    SquareAttack,
    ZooAttack
)
import numpy as np


def auto_projected_cross_entropy(
    estimator, batch_size=32,eps=0.3, eps_step=0.1, norm=np.inf
):
    """
    Create an Auto Projected Gradient Descent attack instance with 
    cross-entropy loss.

    Parameters:
        estimator: The classifier to attack.
        batch_size (int): Batch size for the attack.
        norm: Norm to use for the attack.
        eps (float): Maximum perturbation allowed.
        eps_step (float): Step size of the attack.

    Returns:
        An instance of AutoProjectedGradientDescent.
    """
    return AutoProjectedGradientDescent(
        estimator=estimator, norm=norm, eps=eps, eps_step=eps_step, 
        max_iter=100, targeted=False, nb_random_init=5, batch_size=batch_size,
        loss_type="cross_entropy"
    )


def auto_projected_difference_logits_ratio(
    estimator, batch_size=32, eps=0.3, eps_step=0.1,norm=np.inf
):
    """
    Create an Auto Projected Gradient Descent attack instance with 
    difference logits ratio loss.

    Parameters:
        estimator: The classifier to attack.
        batch_size (int): Batch size for the attack.
        norm: Norm to use for the attack.
        eps (float): Maximum perturbation allowed.
        eps_step (float): Step size of the attack.

    Returns:
        An instance of AutoProjectedGradientDescent.
    """
    return AutoProjectedGradientDescent(
        estimator=estimator, norm=norm, eps=eps, eps_step=eps_step, 
        max_iter=100, targeted=False, nb_random_init=5, batch_size=batch_size,
        loss_type="difference_logits_ratio",
    )


def carlini_L0_attack(
    classifier, batch_size=32,learning_rate=0.01, binary_search_steps=10, max_iter=10, 
    confidence=0.0, targeted=False, 
    initial_const=0.01, mask=None, warm_start=True, max_halving=5, 
    max_doubling=5, verbose=True
):
    """
    Create a Carlini L0 attack instance.

    Parameters:
        classifier: The classifier to attack.
        batch_size (int): Batch size for the attack.
        confidence (float): Confidence parameter.
        targeted (bool): Whether the attack is targeted.
        learning_rate (float): Learning rate for optimization.
        binary_search_steps (int): Number of binary search steps.
        max_iter (int): Maximum number of optimization iterations.
        initial_const (float): Initial constant for optimization.
        mask: Mask for the attack.
        warm_start (bool): Whether to use warm-starting.
        max_halving (int): Maximum number of times to halve the constant.
        max_doubling (int): Maximum number of times to double the constant.
        verbose (bool): Whether to display verbose output.

    Returns:
        An instance of CarliniL0Method.
    """
    return CarliniL0Method(
        classifier, confidence, targeted, learning_rate, binary_search_steps,
        max_iter, initial_const, mask, warm_start, max_halving, max_doubling,
        batch_size, verbose
    )


def carlini_L2_attack(
    classifier, batch_size=32, 
    learning_rate=0.01, binary_search_steps=10, max_iter=10, 
    confidence=0.0, targeted=False, 
    initial_const=0.01, max_halving=5, max_doubling=5, verbose=True
):
    """
    Create a Carlini L2 attack instance.

    Parameters:
        classifier: The classifier to attack.
        batch_size (int): Batch size for the attack.
        confidence (float): Confidence parameter.
        targeted (bool): Whether the attack is targeted.
        learning_rate (float): Learning rate for optimization.
        binary_search_steps (int): Number of binary search steps.
        max_iter (int): Maximum number of optimization iterations.
        initial_const (float): Initial constant for optimization.
        max_halving (int): Maximum number of times to halve the constant.
        max_doubling (int): Maximum number of times to double the constant.
        verbose (bool): Whether to display verbose output.

    Returns:
        An instance of CarliniL2Method.
    """
    return CarliniL2Method(
        classifier, confidence, targeted, learning_rate, binary_search_steps,
        max_iter, initial_const, max_halving, max_doubling, batch_size, 
        verbose,
    )


def carlini_Linf_attack(
    classifier, batch_size=32, learning_rate=0.01,  max_iter=10, confidence=0.0, targeted=False, 
    decrease_factor=0.9, initial_const=1e-05, 
    largest_const=20.0, const_factor=2.0, verbose=True
):
    """
    Create a Carlini Linf attack instance.

    Parameters:
        classifier: The classifier to attack.
        batch_size (int): Batch size for the attack.
        confidence (float): Confidence parameter.
        targeted (bool): Whether the attack is targeted.
        learning_rate (float): Learning rate for optimization.
        max_iter (int): Maximum number of optimization iterations.
        decrease_factor (float): Factor for decreasing the constant.
        initial_const (float): Initial constant for optimization.
        largest_const (float): Maximum constant for optimization.
        const_factor (float): Factor for modifying the constant.
        verbose (bool): Whether to display verbose output.

    Returns:
        An instance of CarliniLInfMethod.
    """
    return CarliniLInfMethod(
        classifier, confidence, targeted, learning_rate, max_iter, 
        decrease_factor, initial_const, largest_const, const_factor, 
        batch_size, verbose,
    )


def deep_fool_attack(
    classifier, batch_size=32, max_iter=100, epsilon=1e-06, nb_grads=10, 
    verbose=True
):
    """
    Create a Deep Fool attack instance.

    Parameters:
        classifier: The classifier to attack.
        batch_size (int): Batch size for the attack.
        max_iter (int): Maximum number of iterations.
        epsilon (float): Perturbation size.
        nb_grads (int): Number of gradients to compute.
        verbose (bool): Whether to display verbose output.

    Returns:
        An instance of DeepFool.
    """
    return DeepFool(
        classifier, max_iter, epsilon, nb_grads, batch_size, verbose
    )



def pixel_attack(
        classifier, max_iter=100,th=None, es=1,  targeted=False, verbose=True
    ):
    """
    Create a Pixel Attack instance.

    Parameters:
        classifier: The classifier to attack.
        th: Threshold for attack.
        es (int): Early stopping criterion.
        max_iter (int): Maximum number of iterations.
        targeted (bool): Whether the attack is targeted.
        verbose (bool): Whether to display verbose output.

    Returns:
        An instance of PixelAttack.
    """
    return PixelAttack(classifier, th, es, max_iter, targeted, verbose)


def square_attack(
    estimator, batch_size=32, max_iter=100, norm=np.inf, adv_criterion=None, loss=None,
 eps=0.3, p_init=0.8, nb_restarts=1, verbose=True
):
    """
    Create a Square Attack instance.

    Parameters:
        estimator: The estimator to attack.
        batch_size (int): Batch size for the attack.
        norm: Norm to use for the attack.
        adv_criterion: Adversarial criterion for the attack.
        loss: Loss function for the attack.
        max_iter (int): Maximum number of iterations.
        eps (float): Maximum perturbation allowed.
        p_init (float): Initial perturbation scaling factor.
        nb_restarts (int): Number of restarts for the attack.
        verbose (bool): Whether to display verbose output.

    Returns:
        An instance of SquareAttack.
    """
    return SquareAttack(
        estimator, norm, adv_criterion, loss, max_iter, eps, p_init, 
        nb_restarts, batch_size, verbose
    )


def zoo_attack(
    classifier, batch_size=32, learning_rate=0.01, max_iter=10, binary_search_steps=1, 
    confidence=0.0, targeted=False, 
    initial_const=0.001, abort_early=True, use_resize=True, 
    use_importance=True, nb_parallel=128, variable_h=0.0001, verbose=True,
):
    """
    Create a Zoo Attack instance.

    Parameters:
        classifier: The classifier to attack.
        batch_size (int): Batch size for the attack.
        confidence (float): Confidence parameter.
        targeted (bool): Whether the attack is targeted.
        learning_rate (float): Learning rate for optimization.
        max_iter (int): Maximum number of optimization iterations.
        binary_search_steps (int): Number of binary search steps.
        initial_const (float): Initial constant for optimization.
        abort_early (bool): Whether to abort early during optimization.
        use_resize (bool): Whether to use resize during optimization.
        use_importance (bool): Whether to use importance during optimization.
        nb_parallel (int): Number of parallel threads.
        variable_h (float): Variable for determining step size.
        verbose (bool): Whether to display verbose output.

    Returns:
        An instance of ZooAttack.
    """
    return ZooAttack(
        classifier, confidence, targeted, learning_rate, max_iter,
        binary_search_steps, initial_const, abort_early, use_resize, 
        use_importance, nb_parallel, batch_size, variable_h, verbose
    )
