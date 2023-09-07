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
    import importlib
    attack_list_estimator=["FastGradientMethod"]
    model_name="art.attacks.evasion"
    m=importlib.import_module(model_name)
    for attack_name in attack_list_estimator:
        c = getattr(m, attack_name)
        attack=c(estimator=PyTorchClassifier)
        batch = next(iter(dataloader_test))

        X, y = batch

        X_advx = attack.generate(x=X.numpy())
        
        dataset_advx = TensorDataset(torch.Tensor(X_advx), y)
        
        dataloader_advx = DataLoader(dataset_advx, batch_size=batch_size_attack, shuffle=False, num_workers=num_threads_attack)
        acc_advx = evaluation(model, dataloader_advx, device)

        print('{} tested, Adversarial examples accuracy: {:.2f}'.format(attack_name,acc_advx * 100))
    attack_list_classifier=["DeepFool"]
    for attack_name in attack_list_classifier:
        c = getattr(m, attack_name)
        attack=c(classifier=PyTorchClassifier)
        batch = next(iter(dataloader_test))

        X, y = batch

        X_advx = attack.generate(x=X.numpy())
        
        dataset_advx = TensorDataset(torch.Tensor(X_advx), y)
        
        dataloader_advx = DataLoader(dataset_advx, batch_size=batch_size_attack, shuffle=False, num_workers=num_threads_attack)
        acc_advx = evaluation(model, dataloader_advx, device)

        print('{} tested, Adversarial examples accuracy: {:.2f}'.format(attack_name,acc_advx * 100))


    
    return result
