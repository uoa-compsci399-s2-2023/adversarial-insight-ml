"""
dynamic.py

This module provides the decide_attack function which will decide the 
next attack to be applied and its parameter.
"""


from aiml.attack.adversarial_attacks import (
    auto_projected_cross_entropy,
    auto_projected_difference_logits_ratio,
    carlini_L0_attack,
    carlini_L2_attack,
    carlini_Linf_attack,
    deep_fool_attack,
    pixel_attack,
    square_attack,
    zoo_attack,
)


def decide_attack(
    result_list,
    attack_para_list=[
    
    ],
    now_time="0",
    ori_acc=0.9
):
    """
    Write the results of the previous attack to a text file and determine 
    the next attack and its parameters based on attack history.

    Parameters:
        result_list: A list where the first element is the overall mark, and the subsequent 
            elements are lists containing the history of previous attacks.
            Sublists stores the attack number, parameter number, and accuracy.
       attack_para_list: list that store parameter for each attack
       now_time(string):program start time
       ori_acc(float): original accuracy(accuracy that test clean image by the model)

    Returns:
        next_attack_number (int): The number of the next attack 
            (could be the same or the next one in the attack_method_list).
        next_parameter_number (int): The number of the next parameter.
        continue_testing (bool): Whether to continue testing attacks or not.
        
    """

    attack_method_list = [
        [
            0,
            auto_projected_cross_entropy,
            attack_para_list[0],
            "auto_projected_cross_entropy",
            ["eps","batch",  "eps_step"],
        ],
        [
            1,
            auto_projected_difference_logits_ratio,
            attack_para_list[1],
            "auto_projected_difference_logits_ratio",
            ["eps","batch",  "eps_step"],
        ],
        [
            2,
            carlini_L0_attack,
            attack_para_list[2],
            "carlini_L0_attack",
            [   "confidence",
                "batch",
                "learning_rate",
                
            ],
        ],
        [
            3,
            carlini_L2_attack,
            attack_para_list[3],
            "carlini_L2_attack",
            [
                "confidence",
                "batch",
                "learning_rate",
                
            ],
        ],
        [
            4,
            carlini_Linf_attack,
            attack_para_list[4],
            "carlini_Linf_attack",
            [
                "confidence",
                "batch",
                "learning_rate",
                
            ],
        ],
        [
            5,
            deep_fool_attack,
            attack_para_list[5],
            "deep_fool_attack",
            ["epsilon","batch", "max_iter"],
        ],
        [
            6,
            pixel_attack,
            attack_para_list[6],
            "pixel_attack",
            ["max_iter"],
        ],
        [
            7,
            square_attack,
            attack_para_list[7],
            "square_attack",
            ["eps","batch", "max_iter"],
        ],
        [
            8,
            zoo_attack,
            attack_para_list[8],
            "zoo_attack",
            [
                "confidence",
                "batch",
                "learning_rate",
                "max_iter",
                
            ],
        ],
    ]
    
    """
    attack_method_list contains all eight adversarial attack methods used.

    Each entry in the list is a sublist representing an attack method:
    - The first element is the attack number.
    - The second element is the attack function.
    - The third element is a list of parameter combinations.
    - The fourth element is the name of the attack.
    - The fifth element is the parameter name for every combination of parameters.

    For example, consider the 'auto_projected_cross_entropy' attack method:
    - The attack number is 0.
    - The attack function is 'auto_projected_cross_entropy'.
    - Three possible parameter choices exist: batch=16, batch=20, or batch=32.
    """


    if result_list==[]:  # add the first attack to initial result list
        return (
            0,
            0,
            True,
            
        )  # current_attack_n,para,current_attack,b

    
    previous = result_list[-1]  # get information of previous attack result
    previous_attack_n = previous[0]
    previous_acc = previous[2]
    previous_para_n = previous[1]
    file_name="attack_evaluation_result"+str(now_time)+".txt"
    with open(
        file_name, "a" #the result will output to attack evaluation result.txt
    ) as f:  # write the results of previous attack to txt file
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
    If the accuracy of previous attack is small enough, it means that the attack with previous 
    parameters is strong enough for the model, then it skip more strong parameter and test next 
    attack.
    If the previous parameters is the most strongest, test next attack 
    the (overall_mark / (len(result_list)-1)) briefly record the robustness of the model. 
    If it >2 it means that it pass the middle strong attack
    on average. It will skip the weak attack later.
    """

    if (
        previous_acc < ori_acc*0.4
        or previous_para_n +1>= len(attack_method_list[previous_attack_n][2]) 
    ):
        strong=(previous_para_n +1)/len(attack_method_list[previous_attack_n][2])
        out_string=""
        if strong>=0.8 and previous_acc >= ori_acc*0.4:
            out_string="your model is very robust on "+str(attack_method_list[previous_attack_n][3])+"\n"
        elif strong>=0.25:
            out_string="your model is barely robust on "+str(attack_method_list[previous_attack_n][3])+"\n"
        else:
            out_string="your model is not robust on "+str(attack_method_list[previous_attack_n][3])+"\n"
        next_para_n = 0
        with open(
            file_name, "a" #the result will output to attack evaluation result.txt
        ) as f:  # write the results of previous attack to txt file
            f.write(out_string+"\n")
        if previous_attack_n < 8:
            
            next_para_n = 0
            return (
                previous_attack_n + 1,
                next_para_n,
                True,
                
            )
        else:  # all attack are tested and finish
            return (
                0,
                0,
                False,
                
            )

    else:
        return (
            previous_attack_n,
            previous_para_n + 1,
            True,
            
        )
