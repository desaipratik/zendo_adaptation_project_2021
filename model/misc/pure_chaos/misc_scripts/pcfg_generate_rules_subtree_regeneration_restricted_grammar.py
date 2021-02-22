####################### General imports #######################################
import numpy as np
import pandas as pd
import random as rd
import re


####################### Custom imports ########################################
from formal_learning_model import prior, likelihood
from pruned_pcfg import pcfg_generator
from transform_functions import compute_orientation, check_structure

ob = pcfg_generator()                                    # instantiating pcfg_generator


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def resample_rule(parse_tree):
    """ creating a random sub tree from the parse tree that is used as start for the regeneration of a new parse tree """
    key = rd.choice(list(parse_tree.keys()))             # selecting random production from parse tree
    val = rd.choice(parse_tree[key])                     # selecting feature/value of a production
    ind = parse_tree[key].index(val)                     # getting the index of the feature within the production
    sub_tree = {i: parse_tree[i] for i in parse_tree if  # creating sub tree
                list(parse_tree.keys()).index(i) <= list(parse_tree.keys()).index(key)}
    sub_tree[key] = sub_tree[key][:ind + 1]              # appending only those features of a production that do not exceed the index of the selected feature
    return sub_tree

def reproduce_full_prod_dict(prod_d, productions):
    """ reproducing the full parse tree dictionary from its reduced version that is used for subtree regeneration
     NOTE: this is currently inconvenient and needs to be changed"""
    keys = list(prod_d.keys())
    full_prod_dict = ob.pd_options(productions)
    keys_D = list(productions["D"].keys())
    keys_E = list(productions["E"].keys())
    for key in keys:
        if len(key) == 2:
            if key[0] == "D":
                new_key = keys_D[int(key[1])]
                for i in prod_d[key]:
                    full_prod_dict["D"][new_key][i] += 1
            elif key[0] == "E":
                new_key = keys_E[int(key[1])]
                for i in prod_d[key]:
                    full_prod_dict["E"][new_key][i] += 1
        else:
            for i in prod_d[key]:
                if isinstance(full_prod_dict[key], list):
                    full_prod_dict[key][i] += 1
    return full_prod_dict

def reproduce_rule(parse_tree, productions):
    """ reproducing a pcfg rule based on a predetermined parse tree (could be a sub tree of another parse tree)
    NOTE: this function needs to be updated to be more concise """
    recode = parse_tree
    rule = "S"
    bound_vars = []

    i_2 = 0
    i_3 = 0
    i_4 = 0
    i_5 = 0
    i_6 = 0
    i_7 = 0
    i_8 = 0
    i_9 = 0
    i_10 = 0
    i_11 = 0
    i_12 = 0
    i_13 = 0
    i_14 = 0

    while any([i for i in ["S", "A"] if (i in [char for char in rule])]):
        srule = [char for char in rule]
        for i in range(0, len(srule)):

            if srule[i] == "S":
                if "S" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_2 == len(recode["S"]):
                        return rule, bound_vars
                    else:
                        bound_vars.append(["x" + str(i + 1) for i in range(0, len(bound_vars) + 1)])
                        replacements = [re.sub("N", str(len(bound_vars)), i) for i in productions["S"]]
                        srule[i] = replacements[recode["S"][i_2]]
                        rule = "".join(srule)
                        i_2 += 1

            if srule[i] == "A":
                replacements = productions["A"]
                if "A" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_3 == len(recode["A"]):
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["A"][i_3]]
                        rule = "".join(srule)
                        i_3 += 1

    while any([i for i in ["B", "C", "D", "E", "G", "H", "I", "J", "K", "L", "M"] if (i in [char for char in rule])]):
        srule = [char for char in rule]
        for i in range(0, len(srule)):
            if srule[i] == 'B':
                replacements = productions["B"]
                if "B" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_4 == len(recode["B"]):
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["B"][i_4]]
                        rule = "".join(srule)
                        i_4 += 1

            if srule[i] == 'C':
                if len(bound_vars) == 1:
                    replacements = [re.sub("N", "1", i) for i in productions["C"]]
                    reps = np.random.choice(np.arange(0, len(bound_vars)), 2)
                    replacements = [re.sub("O", str(reps[1] + 1), i) for i in replacements]
                elif len(bound_vars) > 1:
                    reps = np.random.choice(np.arange(0, len(bound_vars)), 2)
                    replacements = [re.sub("N", str(reps[0] + 1), i) for i in productions["C"]]
                    replacements = [re.sub("O", str(reps[1] + 1), i) for i in replacements]
                if "C" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_5 == len(recode["C"]):
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["C"][i_5]]
                        rule = "".join(srule)
                        i_5 += 1

            if srule[i] == 'D':
                if "D" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_6 == len(recode["D"]):
                        return rule, bound_vars
                    else:
                        feature = list(productions["D"].keys())[recode["D"][i_6]]
                nest_key = "D" + str(recode["D"][i_6])
                if nest_key not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_6 == len(recode["D"]):
                        return rule, bound_vars
                    else:
                        id_1 = recode["D" + str(recode["D"][i_6])]
                        value = productions["D"][feature][id_1[0]]
                        replacement = str(value) + "," + "'" + feature + "'"
                        srule[i] = replacement
                        rule = "".join(srule)
                        i_6 += 1

            if srule[i] == 'E':
                if "E" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_7 == len(recode["E"]):
                        return rule, bound_vars
                    else:
                        feature = list(productions["E"].keys())[recode["E"][i_7]]
                nest_key = "E" + str(recode["E"][i_7])
                if nest_key not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_7 == len(recode["E"]):
                        return rule, bound_vars
                    else:
                        ie_1 = recode["E" + str(recode["E"][i_7])]
                        value = productions["E"][feature][ie_1[0]]
                        replacement = str(value) + "," + "'" + feature + "'"
                        srule[i] = replacement
                        rule = "".join(srule)
                        i_7 += 1

            if srule[i] == 'G':
                replacements = productions["G"]
                if "G" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_8 == len(recode["G"]):
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["G"][i_8]]
                        rule = "".join(srule)
                        i_8 += 1

            if srule[i] == 'H':
                replacements = productions["H"]
                if "H" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_9 == len(recode["H"]):
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["H"][i_9]]
                        rule = "".join(srule)
                        i_9 += 1

            if srule[i] == 'I':
                replacements = productions["I"]
                if "I" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_10 == len(recode["I"]):
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["I"][i_10]]
                        rule = "".join(srule)
                        i_10 += 1

            if srule[i] == 'J':
                replacements = productions["J"]
                if "J" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_11 == len(recode["J"]):
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["J"][i_11]]
                        rule = "".join(srule)
                        i_11 += 1

            if srule[i] == 'K':
                replacements = productions["K"]
                if "K" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_12 == len(recode["K"]):
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["K"][i_12]]
                        rule = "".join(srule)
                        i_12 += 1

            if srule[i] == 'L':
                replacements = productions["L"]
                if "L" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_13 == len(recode["L"]):
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["L"][i_13]]
                        rule = "".join(srule)
                        i_13 += 1

            if srule[i] == 'M':
                replacements = productions["M"]
                if "M" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_14 == len(recode["M"]):
                        return rule, bound_vars
                    else:
                        srule[i] = str(replacements[recode["M"][i_14]])
                        rule = "".join(srule)
                        i_14 += 1

    # reproduces a rule based on the parse tree and returns the rule as well as its bound variables
    return [rule, bound_vars]

def mh_mcmc_sampler_res(productions,
                    data,
                    ground_truth = None,
                    start="S",
                    start_frame=pd.DataFrame({"from": [], "to": [], "toix": [], "li": []}),
                    iter=100,
                    type=None,
                    prod_l=None,
                    prod_d=None,
                    out_penalizer=None,
                    Dwin=None,
                    feat_probs=[[0.333333333,.333333333,.333333333],[0.333333333,.333333333,.333333333],[0],[0],[0],[.25,.25,.25,.25],[.5,.5]]):
    """ key function that samples new rules using subtree regeneration proposals and then applies mh mcmc sampling
    to approximate the posterior. Details of the underlying computations can be found in Goodman, Tenenbaum, Feldman,
    and Griffiths (2008), Cognitive Science 32"""

    t = ob.generate_res(productions, start, [], start_frame,feat_probs=feat_probs,Dwin=Dwin)  # create an initial rule t
    # print('j')
    # print(feat_probs)
    # print(Dwin)
    if type == 'posterior_map':
        t = {"rule": start, "start": start_frame, "prod_l": prod_l, "prod_d": prod_d, "prec": start_frame}


    thin_length = int(iter / 10)  # length of thinned mcmc chain

    df_rules = pd.DataFrame({"rulestring": [""] * thin_length,   # data frame for storing rules and other relevant properties
                             "productions": [None] * thin_length,
                             "prod_l": [None] * thin_length,
                             "prod_d": [None] * thin_length,
                             "li": [None] * thin_length,
                             "n_pds": [None] * thin_length,
                             "results": [None] * thin_length,
                             "gt_trials": [None] * thin_length})

    rule_details = {}  # additional rule details returned in the end

    for i in range(iter):
        # t = ob.generate(productions, start, [], start_frame)
        sub_tree = resample_rule(t["prod_d"])  # new subtree based on parse tree
        len_sub_tree = sum([len(sub_tree[key]) for key in list(sub_tree.keys())])  # len / size of the subtree
        sub_t = reproduce_rule(sub_tree, productions)  # reproducing initial rule based on subtree
        # new proposal rule which is a combination of the subtree and new random productions
        t_prime = ob.generate_res(productions, sub_t[0], sub_t[1], t["prec"].loc[
                                                               0:len_sub_tree - 1:],feat_probs=feat_probs,Dwin=Dwin)  # arguments: 1) sub_t[0] = sub rule, 2) sub_t[1] = sub bound vars,
        # 3) t["prec"].loc[0:len_sub_tree-1:] = properties of rule created by sub tree
        # repeat the above if identical
        while t_prime['rule'] == t['rule']:
            sub_tree = resample_rule(t["prod_d"])  # new subtree based on parse tree
            len_sub_tree = sum([len(sub_tree[key]) for key in list(sub_tree.keys())])  # len / size of the subtree
            sub_t = reproduce_rule(sub_tree, productions)  # reproducing initial rule based on subtree
            t_prime = ob.generate_res(productions, sub_t[0], sub_t[1], t["prec"].loc[
                                                               0:len_sub_tree - 1:],feat_probs=feat_probs,Dwin=Dwin)  # arguments: 1) sub_t[0] = sub rule, 2) sub_t[1] = sub bound vars,
            # print('hi')
        new_sub_tree = t_prime["prod_d"]  # new subtree including novel productions following the previously fixed sub tree
        # new_parse_tree = {**sub_tree,
        #                   **new_sub_tree}  # full parse tree for the new rule (t_prime). NOTE: the sub_tree part is identical to t
        # print('s')
        new_parse_tree = merge_two_dicts(sub_tree, new_sub_tree)

        t_prime[
            "prod_d"] = new_parse_tree     # making sure the productions dictionary equals the full new parse tree and not just the new sub tree

        t_prime["prod_l"] = ob.probs_list(reproduce_full_prod_dict(new_parse_tree, productions))   # same for production list
        # print(t_prime['rule'])


        # computing acceptance probability of t_prime based on equation 17 in Goodman, Tenenbaum, Feldman, and Griffiths (2008), Cognitive Science 32

        # 1) computing prior probabilities for t and t prime
        p = prior()  # see formal learning model for details on class prior
        p_t = p.prod_product(t["prod_l"])
        p_t_prime = p.prod_product(t_prime["prod_l"])

        # 2) computing number of non-terminal symbols in t and t_prime
        nt_t = sum([sum(nt) for nt in t["prod_l"]])
        nt_t_prime = sum([sum(nt) for nt in t_prime["prod_l"]])

        # 3) computing the likelihood for t and t prime (still need to allow comparison with ground truth, for now deterministic)

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
                # print(X)

                res_t.append(eval(t["rule"]))             # evaluating the rules against X
                # print(i_2)
                res_t_prime.append(eval(t_prime["rule"]))
                ground_truth.append(data[i_2]['follow_rule'])  # appending the ground truth for each scene
                ground_truth_trials.append(data[i_2]['follow_rule'])  # appending the ground truth for each scene


        # doing this for the 8 generalizations from the partner
        elif type == 'posterior_map':
            ground_truth = ground_truth # ground truth based on whether scene followed a rule or not
            ground_truth_trials = []
            for i_2 in range(0, len(data)):
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

        # doing this for all 16 data points together
        elif type == 'posterior_all':
            ground_truth_trials = []
            ground_truth_scenes = [] # ground truth based on whether scene followed a rule or not
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

                ground_truth_scenes.append(data[i_2]['follow_rule'])  # appending the ground truth for each scene

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

            # now combining ground truth from initial scenes with generalizations
            ground_truth = ground_truth_scenes + ground_truth



        z_t = [a and b or not a and not b for a, b in zip(ground_truth, res_t)]  # comparing results with ground truth
        z_t_prime = [a and b or not a and not b for a, b in zip(ground_truth, res_t_prime)]



        out_t = len(z_t) - sum(z_t)  # counts the number of false classifications for a rule
        out_t_prime = len(z_t_prime) - sum(z_t_prime)

        # 3b) computing the likelihood of each rule under consideration of the number of outliers
        ll = likelihood()  # see formal learning model for details on likelihood
        ll_t = ll.compute_ll(out_penalizer, out_t)  # note that number 1
        ll_t_prime = ll.compute_ll(out_penalizer, out_t_prime)

        # 4) computing derivation probability of rule1
        deriv_t = np.prod(t["prec"]["li"])
        deriv_t_prime = np.prod(t_prime["prec"]["li"])

        # 5) putting the above together to compute acceptance probability
        p_acceptance = (np.log(ll_t_prime) - np.log(ll_t)) + (np.log(p_t_prime) - np.log(p_t)) + (np.log(nt_t) - np.log(nt_t_prime)) + (np.log(deriv_t_prime) - np.log(deriv_t))

        # print(t['rule'])
        # print((t_prime['rule']))
        # accepting t prime with probability min(1, p_acceptance)
        # if p_acceptance is smaller than 1, accept stochastically based on np.random.rand() which is a number between 0 and 1
        # print(p_acceptance)
        # print(t["rule"])
        # print(t_prime["rule"])
        if np.log(np.random.rand()) < p_acceptance:
            t = t_prime  # if accepted, the new rule becomes the new basis for comparison (t)
            z_t = z_t_prime  # and the results are the results from the new rule
        # if out_t == 0:

        # # appending t
        # if out_t == 0:
        # print(t['rule'])
        # print(t_prime['rule'])
        if i % 10 == 0:
                # print(t['rule'])
            df_rules["rulestring"][int(i/10)] = t["rule"]
            df_rules["li"][int(i/10)] = np.prod(t["prec"]["li"])
            df_rules["productions"][int(i/10)] = t["prod_d"]
            df_rules["n_pds"][int(i/10)] = len(t["prec"])
            df_rules["prod_l"][int(i/10)] = t["prod_l"]
            df_rules["prod_d"][int(i/10)] = t["prod_d"]
            df_rules["results"][int(i/10)] = z_t
            df_rules["gt_trials"][int(i/10)] = ground_truth_trials
            rule_details[str(int(i/10))] = t["prec"]




    # df_rules = df_rules[df_rules.rulestring != ""]





    return [df_rules, rule_details]
