from art.attacks.evasion import *

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
