"""
dynamic.py

This module provides the decide_attack_order function which will decide 
the parameter of attacks to be applied.
"""


from aiml.attack.adversarial_attacks import *


def decide_attack(result_list, classifier):
    attack_method_list = [
        [0, auto_projected_cross_entropy, [[16], [20], [32]]],
        [1, auto_projected_difference_logits_ratio, [[1], [16], [32]]],
        [2, carlini_L0_attack, [[1], [16], [32]]],
        [3, carlini_L2_attack, [[1], [16], [32]]],
        [4, carlini_Linf_attack, [[1], [16], [32]]],
        [5, deep_fool_attack, [[1], [16], [32]]],
        [6, pixel_attack, [[None]]],
        [7, square_attack, [[1], [16], [32]]],
        [8, zoo_attack, [[1], [16], [32]]],
    ]
    if result_list[-1] == 0:
        return (
            0,
            0,
            attack_method_list[0][1](classifier, attack_method_list[0][2][0][0]),
            True,
            0,
        )  # current_attack_n,para,current_attack,b

    overall_mark = result_list[0]
    previous = result_list[-1]
    previous_attack_n = previous[0]
    previous_acc = previous[2]
    previous_para_n = previous[1]
    
    with open("example.txt", "a") as f:
        f.write(str(previous_attack_n))
        f.write("    ")
        f.write(str(previous_para_n))
        f.write("    ")
        f.write(str(previous_acc))
        f.write("\n")

    print(previous_acc)
    if previous_acc < 0.4 or previous_para_n >= len(attack_method_list[previous_attack_n][2])-1:
        if previous_acc < 0.4 :
            overall_mark +=  (len(attack_method_list[previous_attack_n][2])-previous_para_n)
        if previous_attack_n < 8:
            if overall_mark>5 and (overall_mark/len(result_list))>2:
                next_para_n=1
            else:
                next_para_n=0
            return (
                previous_attack_n + 1,
                next_para_n,
                attack_method_list[previous_attack_n + 1][1](
                    classifier, attack_method_list[previous_attack_n + 1][2][0][0]
                ),
                True,
                overall_mark,
            )
        else:     #finish
            return (
                0,
                0,
                attack_method_list[0][1](classifier, attack_method_list[0][2][0][0]),
                False,
                0,
            )

    else:
        return (
            previous_attack_n,
            previous_para_n + 1,
            attack_method_list[previous_attack_n][1](
                classifier,
                attack_method_list[previous_attack_n][2][previous_para_n + 1][0],
            ),
            True,
            overall_mark,
        )
