def black_box_test(black_box_model):
    surrogate_model=create_surrogate_model(black_box_model)
    result=test_all_white_box_attack(surrogate_model)
