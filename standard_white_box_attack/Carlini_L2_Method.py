from art.attacks.evasion import CarliniL2Method

def Carlini_L2_Method(classifier, confidence: float = 0.0, targeted: bool = False, learning_rate: float = 0.01, binary_search_steps: int = 10, max_iter: int = 10, initial_const: float = 0.01, max_halving: int = 5, max_doubling: int = 5, batch_size: int = 1, verbose: bool = True):
    return CarliniL2Method(classifier,confidence,targeted,learning_rate,binary_search_steps,max_iter,initial_const,max_halving,max_doubling,batch_size,verbose)