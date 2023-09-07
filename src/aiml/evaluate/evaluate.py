"""
evaluate.py

This module provides the evaluate function which will call functions to apply all appropriate attacks, 
and determine a risk evaluation based on them (low, medium, high.)
"""

def evaluate(input_model, input_train_data=None, input_test_data=None, input_shape=None, clip_values=None, nb_classes=None, 
             batch_size_attack=64, num_threads_attack=8, batch_size_train=64, batch_size_test=64):
    # Call other modules to perform attacks and receive accuracy
    risk_eval = "HIGH"
    result_list = get_accuracy_results(input_model, input_train_data=None, input_test_data=None, input_shape=None, clip_values=None, nb_classes=None, 
             batch_size_attack=64, num_threads_attack=8, batch_size_train=64, batch_size_test=64)
    
    # Algorithm to weigh and calculate differnt accuracy scores, to give a final risk evaluation 
    
    return risk_eval
