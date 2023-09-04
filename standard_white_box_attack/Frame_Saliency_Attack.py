from art.attacks.evasion import FrameSaliencyAttack

def Frame_Saliency_Attack(classifier, attacker, method: str = 'iterative_saliency', frame_index: int = 1, batch_size: int = 1, verbose: bool = True):
    """
    attacker: EvasionAttack
    """
    return FrameSaliencyAttack(classifier,attacker,method,frame_index,batch_size,verbose)