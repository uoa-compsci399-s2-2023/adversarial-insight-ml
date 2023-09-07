def generate_parameter(input_shape,clip_values,nb_classes,dataset_test,dataloader_test):
    if input_shape == None:
        (x, y) = next(iter(dataset_test))
        input_shape = np.array(x.size())
        print(f'input_shape: {input_shape}')

    if clip_values == None:
        global_min = 9999.
        global_max = 0.
        s = 0
        n = 0
        b = True
        for batch in dataloader_test:
            x, _ = batch
            if b == True:
                print(batch)
                b = False
            global_min = min(torch.min(x).item(), global_min)
            global_max = max(torch.max(x).item(), global_max)
        clip_values = (global_min,global_max)

        print(f'Min: {global_min}, Max: {global_max}')

    if nb_classes == None:
        list1 = []
        for i in range(len(dataset_test)):
            x,y = dataset_test[i]
            if y not in list1:
                list1+=[y]
        nb_classes=len(dataset_train)
    return (input_shape,clip_values,nb_classes)
