from art.attacks.evasion import *
import math

def adversarial_patch(classifier, rotation_max = 22.5, scale_min = 0.1, scale_max = 1.0, learning_rate = 5.0, max_iter = 500, batch_size = 16, patch_shape = None, targeted = True, verbose = True):
    """
    Deep_Fool takes in a classifier and returns an ART DeepFool instance.

    Inputs: 
    classifier: A trained classifier of type CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE.
    rotation_max (float): The maximum rotation applied to random patches. The value is expected to be in the range [0, 180].
    scale_min (float): The minimum scaling applied to random patches. The value should be in the range [0, 1], but less than scale_max.
    scale_max (float): The maximum scaling applied to random patches. The value should be in the range [0, 1], but larger than scale_min.
    learning_rate (float): The learning rate of the optimization.
    max_iter (int): The number of optimization steps.
    batch_size (int): The size of the training batch.
    patch_shape: The shape of the adversarial patch as a tuple of shape (width, height, nb_channels). Currently only supported for TensorFlowV2Classifier. For classifiers of other frameworks the patch_shape is set to the shape of the input samples.
    targeted (bool): Indicates whether the attack is targeted (True) or untargeted (False).
    verbose (bool): Show progress bars.

    Output:
    AdversarialPatch: A AdversarialPatch instance from art.attacks.evasion
    """

    return AdversarialPatch(classifier, rotation_max, scale_min, scale_max, learning_rate, max_iter, batch_size, patch_shape, targeted, verbose)

def carlini_L0_method(classifier, confidence = 0.0, targeted = False, learning_rate = 0.01, binary_search_steps = 10, max_iter = 10, initial_const = 0.01, mask = None, warm_start = True, max_halving = 5, max_doubling = 5, batch_size = 1, verbose = True):
    return CarliniL0Method(classifier,confidence,targeted,learning_rate,binary_search_steps,max_iter, initial_const, mask, warm_start, max_halving, max_doubling, batch_size, verbose)

def carlini_L2_method(classifier, confidence = 0.0, targeted = False, learning_rate = 0.01, binary_search_steps = 10, max_iter = 10, initial_const = 0.01, max_halving = 5, max_doubling = 5, batch_size = 1, verbose = True):
    return CarliniL2Method(classifier,confidence,targeted,learning_rate,binary_search_steps,max_iter,initial_const,max_halving,max_doubling,batch_size,verbose)

def carlini_Linf_method(classifier, confidence = 0.0, targeted = False, learning_rate = 0.01, max_iter = 10, decrease_factor = 0.9, initial_const = 1e-05, largest_const = 20.0, const_factor = 2.0, batch_size = 1, verbose = True):
    return CarliniLInfMethod(classifier, confidence, targeted,learning_rate,max_iter,decrease_factor,initial_const,largest_const,const_factor,batch_size,verbose)

def deep_fool(classifier, max_iter = 100, epsilon = 1e-06, nb_grads = 10, batch_size = 1, verbose = True):
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

def elastic_net(classifier, confidence = 0.0, targeted = False, learning_rate = 0.01, binary_search_steps = 9, max_iter = 100, beta = 0.001, initial_const = 0.001, batch_size = 1, decision_rule = 'EN', verbose = True):
    return ElasticNet(classifier,confidence,targeted,learning_rate,binary_search_steps,max_iter,beta, initial_const,batch_size,decision_rule,verbose)

def feature_adversaries(classifier, delta = None, layer = None, batch_size = 32):
    return FeatureAdversariesNumpy(classifier, delta, layer, batch_size)

def frame_saliency_attack(classifier, attacker, method = 'iterative_saliency', frame_index = 1, batch_size = 1, verbose = True):
    """
    attacker: EvasionAttack
    """
    return FrameSaliencyAttack(classifier,attacker,method,frame_index,batch_size,verbose)

def graphite_blackbox(classifier, noise_size, net_size, heat_patch_size = (4, 4), heat_patch_stride = (1, 1), heatmap_mode = 'Target', tr_lo = 0.65, tr_hi = 0.85, num_xforms_mask = 100, max_mask_size = -1, beta = 1.0, eta = 500, num_xforms_boost = 100, num_boost_queries = 20000, rotation_range = (-30.0, 30.0), dist_range = (0.0, 0.0), gamma_range = (1.0, 2.0), crop_percent_range = (-0.03125, 0.03125), off_x_range = (-0.03125, 0.03125), off_y_range = (-0.03125, 0.03125), blur_kernels = (0, 3), batch_size = 64):
    return (classifier, noise_size, net_size, heat_patch_size, heat_patch_stride, heatmap_mode, tr_lo, tr_hi, num_xforms_mask, max_mask_size, beta, eta, num_xforms_boost, num_boost_queries, rotation_range, dist_range, gamma_range, crop_percent_range, off_x_range, off_y_range, blur_kernels, batch_size)

def graphite_whitebox_pytorch(classifier, net_size, min_tr = 0.8, num_xforms = 100, step_size = 0.0157, steps = 50, first_steps = 500, patch_removal_size = 4, patch_removal_interval = 2, num_patches_to_remove = 4, rand_start_epsilon_range = (-0.03137254901960784, 0.03137254901960784), rotation_range = (-30.0, 30.0), dist_range = (0.0, 0.0), gamma_range = (1.0, 2.0), crop_percent_range = (-0.03125, 0.03125), off_x_range = (-0.03125, 0.03125), off_y_range = (-0.03125, 0.03125), blur_kernels = (0, 3), batch_size = 64):
    return (classifier, net_size, min_tr, num_xforms, step_size, steps, first_steps, patch_removal_size, patch_removal_interval, num_patches_to_remove, rand_start_epsilon_range, rotation_range, dist_range, gamma_range, crop_percent_range, off_x_range, off_y_range, blur_kernels, batch_size)

def high_confidence_low_uncertainty(classifier, conf = 0.95, unc_increase = 100.0, min_val = 0.0, max_val = 1.0, verbose = True):
    return HighConfidenceLowUncertainty(classifier, conf, unc_increase, min_val, max_val, verbose)

def hopskipjump(classifier, batch_size = 64, targeted = False, norm = 2, max_iter = 50, max_eval = 10000, init_eval = 100, init_size = 100, verbose = True):
    return HopSkipJump(classifier, batch_size, targeted, norm, max_iter, max_eval, init_eval, init_size, verbose)

def lowprofool(classifier, n_steps = 100, threshold = 0.5, lambd = 1.5, eta = 0.2, eta_decay = 0.98, eta_min = 1e-07, norm = 2, importance = 'pearson', verbose = False):
    return LowProFool(classifier, n_steps, threshold, lambd, eta, eta_decay, eta_min, norm, importance, verbose)

def newton_fool(classifier, max_iter = 100, eta = 0.01, batch_size = 1, verbose = True):
    return NewtonFool(classifier, max_iter, eta, batch_size, verbose)

def pixel_attack(classifier, th = None, es = 1, max_iter = 100, targeted = False, verbose = False):
    return PixelAttack(classifier, th, es, max_iter, targeted, verbose)

def threshold_attack(classifier, th = None, es = 0, max_iter = 100, targeted = False, verbose = False):
    return ThresholdAttack(classifier, th, es, max_iter, targeted, verbose)

def saliency_map_method(classifier, theta = 0.1, gamma = 1.0, batch_size = 1, verbose = True):
    return SaliencyMapMethod(classifier, theta, gamma, batch_size, verbose)

def shadow_attack(estimator, sigma = 0.5, nb_steps = 300, learning_rate = 0.1, lambda_tv = 0.3, lambda_c = 1.0, lambda_s = 0.5, batch_size = 400, targeted = False, verbose = True):
    return ShadowAttack(estimator, sigma, nb_steps, learning_rate, lambda_tv, lambda_c, lambda_s, batch_size, targeted, verbose)

def simple_blackbox(classifier, attack = 'dct', max_iter = 3000, order = 'random', epsilon = 0.1, freq_dim = 4, stride = 1, targeted = False, batch_size = 1, verbose = True):
    return SimBA(classifier, attack, max_iter, order, epsilon, freq_dim, stride, targeted, batch_size, verbose)

def spatial_transformation(classifier, max_translation = 0.0, num_translations = 1, max_rotation = 0.0, num_rotations = 1, verbose = True):
    return SpatialTransformation(classifier, max_translation, num_translations, max_rotation, num_rotations, verbose)

def targeted_universal_perturbation(classifier, attacker = 'fgsm', attacker_params = None, delta = 0.2, max_iter = 20, eps = 10.0, norm = math.inf):
    return TargetedUniversalPerturbation(classifier, attacker, attacker_params, delta, max_iter, eps, norm)

def universal_perturbation(classifier, attacker = 'deepfool', attacker_params = None, delta = 0.2, max_iter = 20, eps = 10.0, norm = math.inf, batch_size = 32, verbose = True):
    return UniversalPerturbation(classifier, attacker, attacker_params, delta, max_iter, eps, norm, batch_size, verbose)

def virtual_adversarial_method(classifier, max_iter = 10, finite_diff = 1e-06, eps = 0.1, batch_size = 1, verbose = True):
    return VirtualAdversarialMethod(classifier, max_iter, finite_diff, eps, batch_size, verbose)

def zoo_attack(classifier, confidence = 0.0, targeted = False, learning_rate = 0.01, max_iter = 10, binary_search_steps = 1, initial_const = 0.001, abort_early = True, use_resize = True, use_importance = True, nb_parallel = 128, batch_size = 1, variable_h = 0.0001, verbose = True):
    return ZooAttack(classifier, confidence, targeted, learning_rate, max_iter, binary_search_steps, initial_const, abort_early, use_resize, use_importance, nb_parallel, batch_size, variable_h, verbose)
