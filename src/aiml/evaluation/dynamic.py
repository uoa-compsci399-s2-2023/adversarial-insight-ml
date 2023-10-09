"""
dynamic.py

This module provides the decide_attack function which will decide the next attack to be applied and its parameter.

"""


from aiml.attack.adversarial_attacks import *


def decide_attack(result_list,
                attack_para_list=[[[1], [16], [32]],
                      [[1], [16], [32]],
                      [[1], [16], [32]],
                      [[1], [16], [32]],
                      [[1], [16], [32]],
                      [[1], [16], [32]],
                      [[50],[100],[150]],
                      [[1], [16], [32]],
                      [[1], [16], [32]],
                     ]
                 ):
    """
     the function will write the results of previous attack to txt file and decide the next attack to be applied and its parameter based on previous attack history
     arg:
     result_list:the first element is overall mark that briefly record the previous performance as a score. 
                 the left elements are lists contain the history of previous attack. attack number, parameter number, and its accuracy is stored in every list

    return:
                next attack number(int)(may be same or next attack in the attack_method_list),
                next parameter number(int),
                b(boolean): whether continue test attack or not
                overall_mark(int):a score briefly record the previous performance
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
            attack_para_list[0],
            "auto_projected_cross_entropy", 
            ["batch","eps","eps_step"],
        ],
        [
            1, 
            auto_projected_difference_logits_ratio, 
            attack_para_list[1],
            "auto_projected_difference_logits_ratio", 
            ["batch","eps","eps_step"],
        ],
        [
            2, 
            carlini_L0_attack, 
            attack_para_list[2],
            "carlini_L0_attack", 
            ["batch","learning_rate", "binary_search_steps", "max_iter", ],
        ],
        [
            3, 
            carlini_L2_attack, 
            attack_para_list[3],
            "carlini_L2_attack", 
            ["batch","learning_rate", "binary_search_steps", "max_iter",],
        ],
        [
            4, 
            carlini_Linf_attack, 
            attack_para_list[4],
            "carlini_Linf_attack", 
            ["batch","learning_rate",  "max_iter",],
        ],
        [
            5, 
            deep_fool_attack, 
            attack_para_list[5],
            "deep_fool_attack", 
            ["batch","max_iter"],
        ],
        [
            6, 
            attack_para_list[6], 
            "pixel_attack", 
            ["max_iter"],
        ],
        [
            7, 
            square_attack, 
            attack_para_list[7],
            "square_attack", 
            ["batch","max_iter"],
        ],
        [
            8, 
            zoo_attack, 
            attack_para_list[8],
            "zoo_attack", 
            ["batch","learning_rate", "max_iter", "binary_search_steps", ],
        ]
    ]
    if result_list[-1] == 0: #add the first attack to initial result list
        return (
            0,
            0,
            True,
            0,
        )  # current_attack_n,para,current_attack,b

    overall_mark = result_list[0]
    previous = result_list[-1] #get information of previous attack result
    previous_attack_n = previous[0]
    previous_acc = previous[2]
    previous_para_n = previous[1]

    with open("example.txt", "a") as f: #write the results of previous attack to txt file
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

    """
    if the accuracy of previous attack is small enough, it means that the attack with previous parameters is strong enough for the model,
    then it skip more strong parameter and test next attack.
    if the previous parameters is the most strongest, test next attack
    the (overall_mark / (len(result_list)-1)) briefly record the robustness of the model. if it >2 it means that it pass the middle strong attack
    on average. it will skip the weak attack later.
    """


    if (
        previous_acc < 0.4
        or previous_para_n >= len(attack_method_list[previous_attack_n][2]) - 1
    ):
        
        overall_mark += previous_para_n
            
        if previous_attack_n < 8:
            if overall_mark > 5 and (overall_mark / (len(result_list)-1)) > 2:
                next_para_n = 1
            else:
                next_para_n = 0
            return (
                previous_attack_n + 1,
                next_para_n,
                True,
                overall_mark,
            )
        else:  # all attack are tested and finish
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
