"""
test_white_box.py
"""

def test_white_box_attack(attack_method,model,PyTorchClassifier,dataloader_test,batch_size_attack,num_threads_attack,device):
    attack=attack_method(estimator=PyTorchClassifier)
    batch = next(iter(dataloader_test))
    X, y = batch
    X_advx = attack.generate(x=X.numpy())
    dataset_advx = TensorDataset(torch.Tensor(X_advx), y)
    dataloader_advx = DataLoader(dataset_advx, batch_size=batch_size_attack, shuffle=False, num_workers=num_threads_attack)
    acc_advx = evaluation(model, dataloader_advx, device)
    return (acc_advx * 100)
    
def test_all_white_box_attack(model,PyTorchClassifier,dataloader_test,batch_size_attack,num_threads_attack,device):
    attack_method_list = [adversarial_patch,carlini_l0_method,deep_fool]
    accuracy_list=[]
    for attack_method in attack_method_list:
        accuracy_list+=[test_white_box_attack(adversarial_patch,model,PyTorchClassifier,dataloader_test,batch_size_attack,num_threads_attack,device)]
    return result
