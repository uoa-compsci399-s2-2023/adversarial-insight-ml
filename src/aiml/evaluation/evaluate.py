"""
evaluate.py

This module provides the evaluate function which will call functions to 
apply all appropriate attacks, and determine a risk evaluation based on 
them (low, medium, high.)
"""


from aiml.evaluation.get_accuracy_results import get_accuracy_results


# Returns average in a percentage, of given float list
def calculate_average(result_list):
    return (sum(result_list) / len(result_list)) / 100


# Main function for evaluating a model
def evaluate(
    input_model,
    input_train_data=None,
    input_test_data=None,
    input_shape=None,
    clip_values=None,
    nb_classes=None,
    batch_size_attack=64,
    num_threads_attack=8,
    batch_size_train=64,
    batch_size_test=64,
):
    # Call other modules to perform attacks and receive accuracy
    risk_levels = ["LOW", "MEDIUM", "HIGH"]
    risk_eval = risk_levels[2]  # Default risk evaluation is high

    result_list = get_accuracy_results(
        input_model,
        input_train_data,
        input_test_data,
        input_shape,
        clip_values,
        nb_classes,
        batch_size_attack,
        num_threads_attack,
        batch_size_train,
        batch_size_test,
    )

    # Algorithm to determine risk evaluation
    white_box_average = calculate_average(result_list)
    if white_box_average >= 90:
        risk_eval = risk_levels[0]
    elif white_box_average >= 70:
        risk_eval = risk_levels[1]
    else:
        risk_eval = risk_levels[2]

    # Craft summary result string for return
    print(result_list)
    evaluation_summary = (
        " === Risk Evaluation Summary === \n"
        "Average accuracy for white box attacks: {:.1%}\n"
        "Risk level of the model is {}.\n"
    ).format(white_box_average, risk_eval)

    print(evaluation_summary)

    return None
