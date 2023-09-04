from art.attacks.evasion import CarliniLInfMethod

def Carlini_LInf_Method(classifier, confidence: float = 0.0, targeted: bool = False, learning_rate: float = 0.01, max_iter: int = 10, decrease_factor: float = 0.9, initial_const: float = 1e-05, largest_const: float = 20.0, const_factor: float = 2.0, batch_size: int = 1, verbose: bool = True):
    return CarliniLInfMethod(classifier, confidence, targeted,learning_rate,max_iter,decrease_factor,initial_const,largest_const,const_factor,batch_size,verbose)