"""
dynamic.py

This module provides the decide_attack function which will decide the next attack to be applied and its parameter.

"""


from aiml.attack.adversarial_attacks import *


def decide_attack(result_list):
    """
     the function will decide the next attack to be applied and its parameter based on previous attack history
     arg:
     result_list:the first element is overall mark that briefly record the previous performance as a score. 
                 the left elements are lists contain the history of previous attack. attack number, parameter number, and its accuracy is stored in every list

    return:
                int:next attack number(may be same or next attack in the attack_method_list),
                int:next parameter number,
                boolean: whether continue test attack or not
                int:overall_mark:a score briefly record the previous performance
    """

    """
    attack_method_list contains all eight attack methods. 
    every attack has a list to contain the information about the attack. the first element is attack number. second element is attack function. third element is combinations of parameters.
    fourth element is the name of the attack. fifth element is the parameter name for every combination of parameters

    for example, for the auto_projected_cross_entropy attack method, the attack number is 0. the function is auto_projected_cross_entropy.
    three possible parameter choices: batch=16, batch=20 or batch=32
    """
    attack_method_list = [
        [
            0,
            auto_projected_cross_entropy,
            [[16], [20], [32]],
            "auto_projected_cross_entropy",
            ["batch"],
        ],
        [
            1,
            auto_projected_difference_logits_ratio,
            [[1], [16], [32]],
            "auto_projected_difference_logits_ratio",
            ["batch"],
        ],
        [
            2, 
            carlini_L0_attack, 
            [[1], [16], [32]], 
            "carlini_L0_attack", 
            ["batch"]
        ],
        [
            3, 
            carlini_L2_attack, 
            [[1], [16], [32]], 
            "carlini_L2_attack", 
            ["batch"]
        ],
        [
            4, 
            carlini_Linf_attack, 
            [[1], [16], [32]], 
            "carlini_Linf_attack", 
            ["batch"]
        ],
        [
            5, 
            deep_fool_attack, 
            [[1], [16], [32]], 
            "deep_fool_attack", 
            ["batch"]
        ],
        [
            6, 
            pixel_attack, 
            [[None]], 
            "pixel_attack", 
            ["th"]
        ],
        [
            7,
            square_attack, 
            [[1], [16], [32]], 
            "square_attack", 
            ["batch"]
        ],
        [
            8, 
            zoo_attack, 
            [[1], [16], [32]], 
            "zoo_attack", 
            ["batch"]
        ],
    ]
    if result_list[-1] == 0:
        return (
            0,
            0,
            True,
            0,
        )  # current_attack_n,para,current_attack,b

    overall_mark = result_list[0]
    previous = result_list[-1]
    previous_attack_n = previous[0]
    previous_acc = previous[2]
    previous_para_n = previous[1]

    with open("example.txt", "a") as f:
        f.write(attack_method_list[previous_attack_n][3])
        f.write("    ")
        for i in range(len(attack_method_list[previous_attack_n][2][previous_para_n])):
            f.write(attack_method_list[previous_attack_n][4][i])
            f.write("    ")
            f.write(str(attack_method_list[previous_attack_n][2][previous_para_n][i]))
            f.write("    ")
        f.write("accuracy:")
        f.write(str(previous_acc))
        f.write("\n")

    print(previous_acc)
    if (
        previous_acc < 0.4
        or previous_para_n >= len(attack_method_list[previous_attack_n][2]) - 1
    ):
        if previous_acc < 0.4:
            overall_mark += (
                len(attack_method_list[previous_attack_n][2]) - previous_para_n
            )
        if previous_attack_n < 8:
            if overall_mark > 5 and (overall_mark / len(result_list)) > 2:
                next_para_n = 1
            else:
                next_para_n = 0
            return (
                previous_attack_n + 1,
                next_para_n,
                True,
                overall_mark,
            )
        else:  # finish
            return (
                0,
                0,
                False,
                overall_mark,
            )

    else:
        return (
            previous_attack_n,
            previous_para_n + 1,
            True,
            overall_mark,
        )
