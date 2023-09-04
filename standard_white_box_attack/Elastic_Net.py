from art.attacks.evasion import ElasticNet

def Elastic_Net(classifier, confidence: float = 0.0, targeted: bool = False, learning_rate: float = 0.01, binary_search_steps: int = 9, max_iter: int = 100, beta: float = 0.001, initial_const: float = 0.001, batch_size: int = 1, decision_rule: str = 'EN', verbose: bool = True):
    return ElasticNet(classifier,confidence,targeted,learning_rate,binary_search_steps,max_iter,beta, initial_const,batch_size,decision_rule,verbose)