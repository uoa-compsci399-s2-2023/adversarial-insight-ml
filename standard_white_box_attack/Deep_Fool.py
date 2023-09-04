from art.attacks.evasion import DeepFool

def Deep_Fool(classifier, max_iter = 100, epsilon = 1e-06, nb_grads = 10, batch_size = 1, verbose = True):
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

    d = DeepFool(classifier, max_iter, epsilon, nb_grads, batch_size, verbose)
    return d