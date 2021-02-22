"""
Created on Sun Mar 15 15:01:21 2020

@author: jan-philippfranken
"""
###############################################################################
#################### mcmc and support functions for exp 1 #####################
###############################################################################


####################### General imports #######################################
import numpy as np
import pandas as pd
import random as rd
import re
import json
import ast

####################### Custom imports ########################################
from formal_learning_model import prior, likelihood
from pcfg_generator import pcfg_generator
from tree_regrower import tree_regrower
from reverse_rule import reverse_rule
from transform_functions import compute_orientation, check_structure
from tree_surgery import tree_surgery

Z = pcfg_generator()  # instantiating grammar generator (Z is an arbitrary choice, the letter G is already used in the grammar...)
tg = tree_regrower()  # instantiating tree regrower
ts = tree_surgery()
rr = reverse_rule()


####################### Generic MCMC Sampler #########################################
def mcmc_sampler(productions,
                 replacements,
                 data,
                 row=1,
                 ground_truth_labels = None,
                 start="S",
                 start_frame=pd.DataFrame({"from": [], "to": [], "toix": [], "li": []}),
                 bv = None,
                 iter=10,
                 type=None,
                 prod_l=None,
                 prod_d=None,
                 out_penalizer=None,
                 Dwin=None,
                 feat_probs=[[0.333333333,.333333333,.333333333],[0.333333333,.333333333,.333333333],[0],[0],[0],[.25,.25,.25,.25],[.5,.5]]):

    """ key function that samples new rules using subtree regeneration proposals and then applies mh mcmc sampling
    to approximate the posterior."""
    ll = likelihood()
    # generate start t

    if type == 'prior_label':
        # print('hihi')
        t = {"rule": start, "prec": rr.get_prec_recursively(rr.string_to_list(start)), 'bv': bv, 'prod_d': 'cat','prod_l': 'cat'}
#         # print(t)
    else:
        t = Z.generate_res(productions=productions, start=start,prec=start_frame,bound_vars=[],feat_probs=feat_probs,Dwin=Dwin)  # create an initial rule t
#         # print('t')
#         # print(t)


    thin_length = 10000     #  defines the length of the initial data frame, if thinning was used, this would be iter/thinning steps. for now it does not matter

    df_rules = pd.DataFrame({"rulestring": [""] * thin_length,   # data frame for storing rules and other relevant properties
                             "productions": [None] * thin_length,
                             "prod_l": [None] * thin_length,
                             "prod_d": [None] * thin_length,
                             "li": [None] * thin_length,
                             "n_pds": [None] * thin_length,
                             "results": [None] * thin_length,
                             "bv": [None] * thin_length,
                             "gt_trials": [None] * thin_length})

    rule_details = {}  # storing additional rule details
#     # print('hi')
    for i in range(iter):
#         # print('hi')
        if i == int(iter * .5):       # after having completed 50% of iterations, the temperature decreases (out_penalizer is an inverse temp parameter)
            out_penalizer = out_penalizer * 2
#         # print('start')
#         # print(t['rule'])
        # now generating proposal
#         # print(t)
        t_prime_info = tg.regrow_tree(t, productions, replacements) # details needed to generate a proposal
#         # print(type)
#         # print(t_prime_info["t_prime_rule"])
        t_prime = Z.generate_res(productions, t_prime_info["t_prime_rule"],
                             bound_vars=t_prime_info["t_prime_bv"], prec=t_prime_info["t_prime_prec"])

#         # print('info above rule below')
#         # print(t_prime['rule'])

        t_prime['prec'] = rr.get_prec_recursively(rr.string_to_list(t_prime['rule']))

        out_t_process = 0
        out_t_prime_process = 0

        # 3) computing the likelihood for t and t prime
        # 3a) first need to evaluate the rules and check how many outliers they have based on ground truth
        global X          # evaluating rule (requires global X)
        res_t = []        # results for each rule and scenes for t
        res_t_prime = []  # results for each rule and scenes for t prime

        # doing this for the 8 initial scenes
        if type == 'prior':
            ground_truth = [] # ground truth based on whether scene followed a rule or not
            ground_truth_trials = []
            for i_2 in range(0, len(data)):   # first looping over all 8 scenes
                X = []                        # for each scene, X will include the objects of the scene
                for i_3 in range(0, len(data[i_2]['ids'])):  # now looping over the number of objects in each scene
                    object = {"id": data[i_2]['ids'][i_3], "colour":  data[i_2]['colours'][i_3], "size":  data[i_2]['sizes'][i_3], "xpos":  int(np.round(data[i_2]['xpos'][i_3])),
                         "ypos":  int(np.round(data[i_2]['ypos'][i_3])), "rotation":  np.round(data[i_2]['rotations'][i_3],1), "orientation":  data[i_2]['orientations'][i_3],
                         "contact":  data[i_2]['contact'][i_3], "grounded":  data[i_2]['grounded'][i_3]}
                    X.append(object)   # appending each object to X
                res_t.append(eval(t["rule"]))             # evaluating the rules against X
                res_t_prime.append(eval(t_prime["rule"]))
                ground_truth.append(data[i_2]['follow_rule'])  # appending the ground truth for each scene
                ground_truth_trials.append(data[i_2]['follow_rule'])  # appending the ground truth for each scene


        elif type == 'prior_label':
            ground_truth = []
            for i_2 in range(0, len(data)):
                X = []
                for i_3 in range(0, len(data[i_2]['ids'])):  # looping over the number of objects in each scene
                    contact = check_structure(data[i_2]['contact'], i_3)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
                    # getting the properties for each object (triangle / cone) in the scene
                    object = {"id": data[i_2]['ids'][i_3], "colour":  data[i_2]['colours'][i_3] , "size":  data[i_2]['sizes'][i_3], "xpos":  int(np.round(data[i_2]['xpos'][i_3])),
                    "ypos": int(np.round(data[i_2]['ypos'][i_3])), "rotation":  np.round(data[i_2]['rotations'][i_3],1), "orientation":  compute_orientation(data[i_2]['rotations'][i_3])[0],
                    "grounded":  data[i_2]['grounded'][i_3], "contact":  contact}
                    X.append(object)   # appending the object to X which includes all objects for a scene
                res_t.append(eval(t["rule"]))
                ground_truth.append(data[i_2]['follow_rule'])
                res_t_prime.append(eval(t_prime["rule"]))

            z_t = [a and b or not a and not b for a, b in zip(ground_truth, res_t)]  # comparing results with ground truth
            out_t = len(z_t) - sum(z_t)  # counts the number of false classifications for a rule (outliers)
            ll_t = ll.compute_ll(out_penalizer,out_t)  # note that number 1

            print(out_t)
            print(ll_t)
            print('testing the wsls stuff')
            # # print(t['rule'])
            # # print('hi')

            return ll_t



        # doing this for all 16 data points together
        elif type == 'posterior_all':
            ground_truth_trials_init = []
            ground_truth_trials_rev = []
            # first for initial scenes
            for i_2 in range(0, len(data[:8])):   # first looping over all 8 scenes
                X = []                        # for each scene, X will include the objects of the scene
                for i_3 in range(0, len(data[i_2]['ids'])):  # now looping over the number of objects in each scene
                    object = {"id": data[i_2]['ids'][i_3], "colour":  data[i_2]['colours'][i_3], "size":  data[i_2]['sizes'][i_3], "xpos": int(np.round(data[i_2]['xpos'][i_3])),
                         "ypos": int(np.round(data[i_2]['ypos'][i_3])), "rotation":  np.round(data[i_2]['rotations'][i_3],1), "orientation":  data[i_2]['orientations'][i_3],
                         "contact":  data[i_2]['contact'][i_3], "grounded":  data[i_2]['grounded'][i_3]}
                    X.append(object)   # appending each object to X
                res_t.append(eval(t["rule"]))             # evaluating the rules against X
                res_t_prime.append(eval(t_prime["rule"]))
                ground_truth_trials_init.append(data[i_2]['follow_rule'])  # appending the ground truth for each scene

            # then for generalizations
            for i_4 in range(8, len(data)):
                X = []
                for i_5 in range(0, len(data[i_4]['ids'])):  # looping over the number of objects in each scene
                    contact = check_structure(data[i_4]['contact'], i_5)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
                    # getting the properties for each object (triangle / cone) in the scene
                    object = {"id": data[i_4]['ids'][i_5], "colour":  data[i_4]['colours'][i_5] , "size":  data[i_4]['sizes'][i_5], "xpos":  int(np.round(data[i_4]['xpos'][i_5])),
                    "ypos": int(np.round(data[i_4]['ypos'][i_5])), "rotation":  np.round(data[i_4]['rotations'][i_5],1), "orientation":  compute_orientation(data[i_4]['rotations'][i_5])[0],
                    "grounded":  data[i_4]['grounded'][i_5], "contact":  contact}
                    X.append(object)   # appending the object to X which includes all objects for a scene
                res_t.append(eval(t["rule"]))             # evaluating the rules against X
                res_t_prime.append(eval(t_prime["rule"]))
                ground_truth_trials_rev.append(data[i_4]['follow_rule'])

            # now combining ground truth from initial scenes with generalizations
            ground_truth = ground_truth_trials_init + ground_truth_trials_rev


            # z_t_process = [a and b or not a and not b for a, b in zip(ground_truth_labels, res_t_process)]  # comparing results with ground truth
            # z_t_prime_process = [a and b or not a and not b for a, b in zip(ground_truth_labels, res_t_prime_process)]

            # out_t_process = len(z_t_process) - sum(z_t_process)  # counts the number of false classifications for a rule
            # out_t_prime_process = len(z_t_prime_process) - sum(z_t_prime_process)




        z_t = [a and b or not a and not b for a, b in zip(ground_truth, res_t)]  # comparing results with ground truth
        z_t_prime = [a and b or not a and not b for a, b in zip(ground_truth, res_t_prime)]

        out_t = len(z_t) - sum(z_t)  # counts the number of false classifications for a rule (outliers)
        out_t_prime = len(z_t_prime) - sum(z_t_prime)

        # 3b) computing the likelihood of each rule under consideration of the number of outliers
        ll = likelihood()  # see formal learning model for details on likelihood
        ll_t = ll.compute_ll(out_penalizer,out_t)  # note that number 1
        ll_t_prime = ll.compute_ll(out_penalizer,out_t_prime)

        # 4) computing derivation probability of rule1
        deriv_t = np.prod(t["prec"]["li"])
#         # print(t_prime['prec'])
        deriv_t_prime = np.prod(t_prime["prec"]["li"])
#         # print(t_prime)
        # 5) putting the above together to compute acceptance probability(np.log(p_t_prime) - np.log(p_t)) +v
        p_acceptance = (np.log(ll_t_prime) - np.log(ll_t)) + (np.log(deriv_t_prime) - np.log(deriv_t))
        if np.log(np.random.rand()) < p_acceptance:
#             # print(type)


            t = t_prime
            z_t = z_t_prime

        if i >= .8 * iter:            # keeping only final 20% of samples
            t['rule'] = rr.list_to_string(rr.string_to_list(t['rule']))
            df_rules["rulestring"][row] = t["rule"]
            df_rules["li"][int(row)] = np.prod(t["prec"]["li"])
            df_rules["productions"][int(row)] = t["prod_d"]
            df_rules["n_pds"][int(row)] = len(t["prec"])
            df_rules["prod_l"][int(row)] = t["prod_l"]
            df_rules["prod_d"][int(row)] = t["prod_d"]
            df_rules["bv"][int(row)] = t["bv"]
            df_rules["results"][int(row)] = z_t
            df_rules["gt_trials"][int(row)] = ground_truth
            rule_details[str(int(row))] = t["prec"]

            row += 1  # print(row)
            # # print(df_rules)


    df_rules = df_rules.dropna()
#     # print(df_rules)
    return [df_rules, rule_details]

####################### Generic MCMC Sampler #########################################
def mcmc_sampler_map(productions,
                 replacements,
                 data,
                 row=1,
                 ground_truth_labels = None,
                 start="S",
                 start_frame=pd.DataFrame({"from": [], "to": [], "toix": [], "li": []}),
                 iter=10,
                 prod_l=None,
                 prod_d=None,
                 out_penalizer=None,
                 bv = None,
                 test_data = None,
                 Dwin=None,
                 feat_probs=[[0.333333333,.333333333,.333333333],[0.333333333,.333333333,.333333333],[0],[0],[0],[.25,.25,.25,.25],[.5,.5]]):

    """ key function that samples new rules using subtree regeneration proposals and then applies mh mcmc sampling
    to approximate the posterior."""

    # now start is input from previous steps
    t = {"rule": start, "prec": rr.get_prec_recursively(rr.string_to_list(start)), 'bv': bv, 'prod_d': 'cat','prod_l': 'cat'}

    acceptance_probs = []
    t_primes = []

#     # print(t)

    thin_length = 1000     #  defines the length of the initial data frame, if thinning was used, this would be iter/thinning steps. for now it does not matter

    df_rules = pd.DataFrame({"rulestring": [None] * thin_length,
                            'acceptance_probs': [None] * thin_length,
                             'check_val': [None] * thin_length})   # data frame for storing rules and other relevant properties
                             # "productions": [None] * thin_length,
                             # "prod_l": [None] * thin_length,
                             # "prod_d": [None] * thin_length,
                             # "li": [None] * thin_length,
                             # "n_pds": [None] * thin_length,
                             # "results": [None] * thin_length,
                             # "gt_trials": [None] * thin_length})
    rule_details = {}  # storing additional rule details

    for i in range(iter):
        t = {"rule": start, "prec": rr.get_prec_recursively(rr.string_to_list(start)), 'bv': bv, 'prod_d': 'cat','prod_l': 'cat'}
#         # print('start')
#         # print(t['rule'])
        # # now generating proposal
        # number of edits
        edits = np.random.poisson(lam=1) + 1

        t_prime_info = tg.regrow_tree(t, productions, replacements) # details needed to generate a proposal
        t_prime = Z.generate_res(productions, t_prime_info["t_prime_rule"],
                             bound_vars=t_prime_info["t_prime_bv"], prec=t_prime_info["t_prime_prec"])
#         # print('proposal')
#         # print(t_prime)
        t_prime['prec'] = rr.get_prec_recursively(rr.string_to_list(t_prime['rule']))

        # p_t_prime = p.prod_product(t_prime["prod_l"])
        #
        # # 2) computing number of non-terminal symbols in t and t_prime
        # nt_t = sum([sum(nt) for nt in t["prod_l"]])
        # nt_t_prime = sum([sum(nt) for nt in t_prime["prod_l"]])

        # 3) computing the likelihood for t and t prime
        # 3a) first need to evaluate the rules and check how many outliers they have based on ground truth
        global X          # evaluating rule (requires global X)
        res_t = []        # results for each rule and scenes for t
        res_t_prime = []  # results for each rule and scenes for t prime

        # doing this for the 8 generalizations from the partner

        res_t_process = []
        res_t_prime_process = []

        ground_truth = []# ground truth based on whether scene followed a rule or not
        ground_truth_labels = ground_truth_labels

        for i_2 in range(0, len(data[:8])):
            X = []
            for i_3 in range(0, len(data[i_2]['ids'])):  # looping over the number of objects in each scene
                contact = check_structure(data[i_2]['contact'], i_3)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
                # getting the properties for each object (triangle / cone) in the scene
                object = {"id": data[i_2]['ids'][i_3], "colour":  data[i_2]['colours'][i_3] , "size":  data[i_2]['sizes'][i_3], "xpos":  int(np.round(data[i_2]['xpos'][i_3])),
                "ypos": int(np.round(data[i_2]['ypos'][i_3])), "rotation":  np.round(data[i_2]['rotations'][i_3],1), "orientation":  compute_orientation(data[i_2]['rotations'][i_3])[0],
                "grounded":  data[i_2]['grounded'][i_3], "contact":  contact}
                X.append(object)   # appending the object to X which includes all objects for a scene
            res_t.append(eval(t["rule"]))             # evaluating the rules against X
            res_t_prime.append(eval(t_prime["rule"]))
            ground_truth.append(data[i_2]['follow_rule'])


        for i_4 in range(8, len(data)):
            X = []
            for i_5 in range(0, len(data[i_4]['ids'])):  # looping over the number of objects in each scene
                contact = check_structure(data[i_4]['contact'], i_5)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
                # getting the properties for each object (triangle / cone) in the scene
                object = {"id": data[i_4]['ids'][i_5], "colour":  data[i_4]['colours'][i_5] , "size":  data[i_4]['sizes'][i_5], "xpos":  int(np.round(data[i_4]['xpos'][i_5])),
                "ypos": int(np.round(data[i_4]['ypos'][i_5])), "rotation":  np.round(data[i_4]['rotations'][i_5],1), "orientation":  compute_orientation(data[i_4]['rotations'][i_5])[0],
                "grounded":  data[i_4]['grounded'][i_5], "contact":  contact}
                X.append(object)   # appending the object to X which includes all objects for a scene

            res_t_process.append(eval(t["rule"]))             # evaluating the rules against X
            res_t_prime_process.append(eval(t_prime["rule"]))
            # ground_truth_trials_rev.append(data[i_4]['follow_rule'])

        z_t_process = [a and b or not a and not b for a, b in zip(ground_truth_labels, res_t_process)]  # comparing results with ground truth
        z_t_prime_process = [a and b or not a and not b for a, b in zip(ground_truth_labels, res_t_prime_process)]

        out_t_process = len(z_t_process) - sum(z_t_process)  # counts the number of false classifications for a rule
        out_t_prime_process = len(z_t_prime_process) - sum(z_t_prime_process)

        z_t = [a and b or not a and not b for a, b in zip(ground_truth, res_t)]  # comparing results with ground truth
        z_t_prime = [a and b or not a and not b for a, b in zip(ground_truth, res_t_prime)]

        out_t = len(z_t) - sum(z_t)  # counts the number of false classifications for a rule (outliers)
        out_t_prime = len(z_t_prime) - sum(z_t_prime)

        # 3b) computing the likelihood of each rule under consideration of the number of outliers
        ll = likelihood()  # see formal learning model for details on likelihood
        ll_t = ll.compute_ll(out_penalizer, out_t)  # note that number 1
        ll_t_prime = ll.compute_ll(out_penalizer, out_t_prime)

        # 4) computing derivation probability of rule1
        deriv_t = np.prod(t["prec"]["li"])
        deriv_t_prime = np.prod(t_prime["prec"]["li"])

        # 5) putting the above together to compute acceptance probability(np.log(p_t_prime) - np.log(p_t)) +
        # v
        p_acceptance = (np.log(ll_t_prime) - np.log(ll_t)) + (np.log(deriv_t_prime) - np.log(deriv_t))

        acceptance_probs.append(p_acceptance)
        t_primes.append(t_prime)


        gt_test = []
        res_test = []
        # still need to check if new proposal is not a flawed rule that is always true:
        for i_6 in range(0, len(test_data)):
            X = []
            for i_7 in range(0, len(test_data[i_6]['ids'])):  # looping over the number of objects in each scene
                contact = check_structure(test_data[i_6]['contact'], i_7)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
                # getting the properties for each object (triangle / cone) in the scene
                object = {"id": test_data[i_6]['ids'][i_7], "colour":  test_data[i_6]['colours'][i_7] , "size":  test_data[i_6]['sizes'][i_7], "xpos":  int(np.round(test_data[i_6]['xpos'][i_7])),
                "ypos": int(np.round(test_data[i_6]['ypos'][i_7])), "rotation":  np.round(test_data[i_6]['rotations'][i_7],1), "orientation":  compute_orientation(test_data[i_6]['rotations'][i_7])[0],
                "grounded":  test_data[i_6]['grounded'][i_7], "contact":  contact}
                X.append(object)   # appending the object to X which includes all objects for a scene
            # res_t.append(eval(t["rule"]))             # evaluating the rules against X
            res_test.append(eval(t_prime["rule"]))
            gt_test.append(test_data[i_6]['follow_rule'])
#         # print(ground_truth)

        acceptance_probs.append(p_acceptance)
        check_val = 0

        if np.log(np.random.rand()) < p_acceptance:
#             # print(p_acceptance)
            t = t_prime
            z_t = z_t_prime
            check_val = 1
#             # print('accepted')

        # if i >= .99 * iter:
#         #     # print(acceptance_probs)
#         #     # print(t_primes)
        #     max_t_prime = max(acceptance_probs)
        #     max_ind = acceptance_probs.index(max_t_prime)
        #
        #     t_prime = t_primes[max_ind]
        #     if 1 < p_acceptance:
#         #         print('hi')
        #         t = t_prime
#         #         print(t)
        #         z_t = z_t_prime
#         #     print('vat')
#         # print(row)
#         # print('rowabove')
#         # print('finalrule below')
#         # print(t['rule'])
        t['rule'] = rr.list_to_string(rr.string_to_list(t['rule']))
        df_rules["rulestring"][row] = t["rule"]
        df_rules['acceptance_probs'][row] = p_acceptance
        df_rules['check_val'][row] = check_val
#         # print(df_rules)
        # df_rules["li"][int(row)] = np.prod(t["prec"]["li"])
        # df_rules["productions"][int(row)] = t["prod_d"]
        # df_rules["n_pds"][int(row)] = len(t["prec"])
        # df_rules["prod_l"][int(row)] = t["prod_l"]
        # df_rules["prod_d"][int(row)] = t["prod_d"]
        # df_rules["results"][int(row)] = z_t
        # df_rules["gt_trials"][int(row)] = ground_truth
        # rule_details[str(int(row))] = t["prec"]
        row += 1  # print(row)
    # if type == "posterior_map":
# #     print(t["rule"])
#     #     print("at")
    df_rules = df_rules.dropna()
#     # print('readydf')
#     # print(df_rules)
    return [df_rules, rule_details]



####################### Generic MCMC Sampler #########################################
def mcmc_sampler_map_surgery(productions,
                 replacements,
                 data,
                 row=1,
                 ground_truth_labels = None,
                 start="S",
                 start_frame=pd.DataFrame({"from": [], "to": [], "toix": [], "li": []}),
                 iter=10,
                 prod_l=None,
                 prod_d=None,
                 out_penalizer=None,
                             test_data = None,
                 bv = None,
                 Dwin=None,
                 feat_probs=[[0.333333333,.333333333,.333333333],[0.333333333,.333333333,.333333333],[0],[0],[0],[.25,.25,.25,.25],[.5,.5]]):

    """ key function that samples new rules using subtree regeneration proposals and then applies mh mcmc sampling
    to approximate the posterior."""

    # replacement dictionary
    rep = {"S": ['Z.exists','Z.forall','Z.atleast','Z.atmost','Z.exactly'],
                    "L": ['Z.atleast','Z.atmost','Z.exactly'],
                    "J": ['and_operator', 'or_operator'],
                    "M": [1,2,3],
                    "A": ['Z.exists','Z.forall','Z.atleast','Z.atmost','Z.exactly'],
                    "B": ['Z.equal', 'Z.hor_operator', 'Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater','Z.and_operator','Z.or_operator','Z.not_operator'],
                    "C": ['Z.equal', 'Z.hor_operator', 'Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']}

    rep_new = {"S": ['Z.exists','Z.forall','Z.atleast','Z.atmost','Z.exactly'],
                    "L": ['Z.atleast','Z.atmost','Z.exactly'],
                    "J": ['and_operator', 'or_operator'],
                    "M": [1,2,3],
                    "A": ['Z.exists','Z.forall','Z.atleast','Z.atmost','Z.exactly'],
                    "B": [["'(xN, D)'","'(xN,xO,G)'"], ["'(xN,xO,I)'","'(xN,xO,I)'"], ["'(xN,xO,H)'","'(xN, E)'"], ["'(xN,xO,H)'","'(xN, E)'"], ["'(xN,xO,H)'","'(xN, E)'"], ["'(xN,xO,H)'","'(xN, E)'"],'Z.and_operator','Z.or_operator','Z.not_operator'],
                    "C": ['Z.equal', 'Z.hor_operator', 'Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']}




    # now start is input from previous steps
    t = {"rule": start, "prec": rr.get_prec_recursively(rr.string_to_list(start)), 'bv': bv}
#     # print(t)

    thin_length = 1000     #  defines the length of the initial data frame, if thinning was used, this would be iter/thinning steps. for now it does not matter

    df_rules = pd.DataFrame({"rulestring": [None] * thin_length,
                            'acceptance_probs': [None] * thin_length,
                             'check_val': [None] * thin_length})   # data frame for storing rules and other relevant properties
                             # "productions": [None] * thin_length,
                             # "prod_l": [None] * thin_length,
                             # "prod_d": [None] * thin_length,
                             # "li": [None] * thin_length,
                             # "n_pds": [None] * thin_length,
                             # "results": [None] * thin_length,
                             # "gt_trials": [None] * thin_length})

    rule_details = {}  # storing additional rule details

    acceptance_probs = []
    t_primes = []

    for i in range(iter):

        t = {"rule": start, "prec": rr.get_prec_recursively(rr.string_to_list(start)), 'bv': bv}
#         # print('surgery')
#         # print('start')
#         # print(t['rule'])


        edits = np.random.poisson(lam=1) + 1
        # edits = 1

        t_prime_info = ts.tree_surgery(t, productions, rep, rep_new) # details needed to generate a proposal
#         # print(t_prime_info['t_prime_rule'])
        # t_prime_info['t_prime_prec'] = rr.get_prec_recursively(rr.string_to_list(t_prime_info['t_prime_rule']))
        t_prime = Z.generate_res(productions, t_prime_info["t_prime_rule"],
                             bound_vars=t_prime_info["t_prime_bv"])
#         # print('proposal')
        t_prime['prec'] = rr.get_prec_recursively(rr.string_to_list(t_prime['rule']))
#         # print(t_prime)


        # if edits > 1:
        #     for edit in range(0, edits-1):
        #
        #         t_prime_info = ts.tree_surgery(t_prime, productions, rep, rep_new) # details needed to generate a proposal
#         #     # print(t_prime_info['t_prime_rule'])
        #         # t_prime_info['t_prime_prec'] = rr.get_prec_recursively(rr.string_to_list(t_prime_info['t_prime_rule']))
        #         t_prime = Z.generate_res(productions, t_prime_info["t_prime_rule"],
        #                              bound_vars=t_prime_info["t_prime_bv"])
#         #         # print('proposal')
        #         t_prime['prec'] = rr.get_prec_recursively(rr.string_to_list(t_prime['rule']))

#         # print('proposal')
#         # print(t_prime['rule'])
        # # 1) computing prior probabilities for t and t prime
        # p = prior()  # see formal learning model for details on class prior
        # p_t = p.prod_product(t["prod_l"])
        # p_t_prime = p.prod_product(t_prime["prod_l"])
        #
        # # 2) computing number of non-terminal symbols in t and t_prime
        # nt_t = sum([sum(nt) for nt in t["prod_l"]])
        # nt_t_prime = sum([sum(nt) for nt in t_prime["prod_l"]])

        # 3) computing the likelihood for t and t prime
        # 3a) first need to evaluate the rules and check how many outliers they have based on ground truth
        global X          # evaluating rule (requires global X)
        res_t = []        # results for each rule and scenes for t
        res_t_prime = []  # results for each rule and scenes for t prime

        # doing this for the 8 generalizations from the partner

        res_t_process = []
        res_t_prime_process = []

        ground_truth = []# ground truth based on whether scene followed a rule or not
        ground_truth_labels = ground_truth_labels

        for i_2 in range(0, len(data[:8])):
            X = []
            for i_3 in range(0, len(data[i_2]['ids'])):  # looping over the number of objects in each scene
                contact = check_structure(data[i_2]['contact'], i_3)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
                # getting the properties for each object (triangle / cone) in the scene
                object = {"id": data[i_2]['ids'][i_3], "colour":  data[i_2]['colours'][i_3] , "size":  data[i_2]['sizes'][i_3], "xpos":  int(np.round(data[i_2]['xpos'][i_3])),
                "ypos": int(np.round(data[i_2]['ypos'][i_3])), "rotation":  np.round(data[i_2]['rotations'][i_3],1), "orientation":  compute_orientation(data[i_2]['rotations'][i_3])[0],
                "grounded":  data[i_2]['grounded'][i_3], "contact":  contact}
                X.append(object)   # appending the object to X which includes all objects for a scene
#             # print(X)
#             # print('hi')
            res_t.append(eval(t["rule"]))             # evaluating the rules against X
            res_t_prime.append(eval(t_prime["rule"]))
            ground_truth.append(data[i_2]['follow_rule'])


        for i_4 in range(8, len(data)):
            X = []
            for i_5 in range(0, len(data[i_4]['ids'])):  # looping over the number of objects in each scene
                contact = check_structure(data[i_4]['contact'], i_5)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
                # getting the properties for each object (triangle / cone) in the scene
                object = {"id": data[i_4]['ids'][i_5], "colour":  data[i_4]['colours'][i_5] , "size":  data[i_4]['sizes'][i_5], "xpos":  int(np.round(data[i_4]['xpos'][i_5])),
                "ypos": int(np.round(data[i_4]['ypos'][i_5])), "rotation":  np.round(data[i_4]['rotations'][i_5],1), "orientation":  compute_orientation(data[i_4]['rotations'][i_5])[0],
                "grounded":  data[i_4]['grounded'][i_5], "contact":  contact}
                X.append(object)   # appending the object to X which includes all objects for a scene

            res_t_process.append(eval(t["rule"]))             # evaluating the rules against X
            res_t_prime_process.append(eval(t_prime["rule"]))
            # ground_truth_trials_rev.append(data[i_4]['follow_rule'])

        z_t_process = [a and b or not a and not b for a, b in zip(ground_truth_labels, res_t_process)]  # comparing results with ground truth
        z_t_prime_process = [a and b or not a and not b for a, b in zip(ground_truth_labels, res_t_prime_process)]

        out_t_process = len(z_t_process) - sum(z_t_process)  # counts the number of false classifications for a rule
        out_t_prime_process = len(z_t_prime_process) - sum(z_t_prime_process)

        z_t = [a and b or not a and not b for a, b in zip(ground_truth, res_t)]  # comparing results with ground truth
        z_t_prime = [a and b or not a and not b for a, b in zip(ground_truth, res_t_prime)]

        out_t = len(z_t) - sum(z_t)  # counts the number of false classifications for a rule (outliers)
        out_t_prime = len(z_t_prime) - sum(z_t_prime)

        # 3b) computing the likelihood of each rule under consideration of the number of outliers
        ll = likelihood()  # see formal learning model for details on likelihood
        ll_t = ll.compute_ll(out_penalizer, out_t)  # note that number 1
        ll_t_prime = ll.compute_ll(out_penalizer, out_t_prime)

        # 4) computing derivation probability of rule1
        deriv_t = np.prod(t["prec"]["li"])
        deriv_t_prime = np.prod(t_prime["prec"]["li"])

        # 5) putting the above together to compute acceptance probability(np.log(p_t_prime) - np.log(p_t)) +
        # v
        p_acceptance = (np.log(ll_t_prime) - np.log(ll_t)) + (np.log(deriv_t_prime) - np.log(deriv_t))

        acceptance_probs.append(p_acceptance)
        t_primes.append(t_prime)

        gt_test = []
        res_test = []
        # still need to check if new proposal is not a flawed rule that is always true:
        for i_6 in range(0, len(test_data)):
            X = []
            for i_7 in range(0, len(test_data[i_6]['ids'])):  # looping over the number of objects in each scene
                contact = check_structure(test_data[i_6]['contact'], i_7)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
                # getting the properties for each object (triangle / cone) in the scene
                object = {"id": test_data[i_6]['ids'][i_7], "colour":  test_data[i_6]['colours'][i_7] , "size":  test_data[i_6]['sizes'][i_7], "xpos":  int(np.round(test_data[i_6]['xpos'][i_7])),
                "ypos": int(np.round(test_data[i_6]['ypos'][i_7])), "rotation":  np.round(test_data[i_6]['rotations'][i_7],1), "orientation":  compute_orientation(test_data[i_6]['rotations'][i_7])[0],
                "grounded":  test_data[i_6]['grounded'][i_7], "contact":  contact}
                X.append(object)   # appending the object to X which includes all objects for a scene
            # res_t.append(eval(t["rule"]))             # evaluating the rules against X
            res_test.append(eval(t_prime["rule"]))
            gt_test.append(test_data[i_6]['follow_rule'])

        acceptance_probs.append(p_acceptance)
        check_val = 0
        # print(p_acceptance)
        if np.log(np.random.rand()) < p_acceptance:
            # print(t['rule'])
            check_val = 1
            t = t_prime
            z_t = z_t_prime
            # print(p_acceptance)
            # print('accepted')
            # print('final rule')
            # print(t['rule'])
        #
        #
        # # if i >= .99 * iter:
#         #     # print(acceptance_probs)
#         #     # print(t_primes)
        #     max_t_prime = max(acceptance_probs)
        #     max_ind = acceptance_probs.index(max_t_prime)
        #
        #     t_prime = t_primes[max_ind]
        #     if 1 < p_acceptance:
#         #         print('hi')
        #         t = t_prime
#         #         print(t)
        #         z_t = z_t_prime
#         # print(row)
#         # print('rowabove')
        t['rule'] = rr.list_to_string(rr.string_to_list(t['rule']))
        df_rules["rulestring"][row] = t["rule"]
        df_rules['acceptance_probs'][row] = p_acceptance
        df_rules['check_val'][row] = check_val
#         # print(df_rules)
        # df_rules["li"][int(row)] = np.prod(t["prec"]["li"])
        # df_rules["productions"][int(row)] = t["prod_d"]
        # df_rules["n_pds"][int(row)] = len(t["prec"])
        # df_rules["prod_l"][int(row)] = t["prod_l"]
        # df_rules["prod_d"][int(row)] = t["prod_d"]
        # df_rules["results"][int(row)] = z_t
        # df_rules["gt_trials"][int(row)] = ground_truth
        # rule_details[str(int(row))] = t["prec"]
        row += 1  # print(row)
    # if type == "posterior_map":
#     #     print(t["rule"])
#     #     print("at")

    df_rules = df_rules.dropna()
#     # print('readydf')
#     # print(df_rules)
    return [df_rules, rule_details]






