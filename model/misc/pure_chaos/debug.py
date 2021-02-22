import random as rd
# from mcmc_cond_1 import reproduce_rule
import numpy as np
import pandas as pd
import re

from pruned_pcfg import pcfg_generator
ob = pcfg_generator()
####################### grammar ##############################################
S = ['ob.exists(lambda xN: A, X)', 'ob.forall(lambda xN: A, X)', 'L(lambda xN: A, M, X)']
A = ['B', 'S']
B = ['C', 'J(B,B)', 'ob.not_operator(B)']
C = ['ob.equal(xN, D)', 'K(xN, E)', 'ob.equal(xN, xO, G)', 'K(xN, xO, H)', 'ob.hor_operator(xN,xO,I)']
D = {"colour": ["'red'", "'blue'", "'green'"], "size": [1, 2, 3], "xpos": np.arange(9),
     "ypos": np.arange(2, 6), "rotation": np.arange(0, 6.5, 0.5),
     "orientation": ["'upright'", "'lhs'", "'rhs'", "'strange'"], "grounded": ["'no'", "'yes'"]}
E = {"size": [1, 2, 3], "xpos": np.arange(9), "ypos": np.arange(2, 6), "rotation": np.arange(0, 6.3, 0.1)}
G = ["'colour'", "'size'", "'xpos'", "'ypos'", "'rotation'", "'orientation'", "'grounded'"]
H = ["'size'", "'xpos'", "'ypos'", "'rotation'"]
I = ["'contact'"]
J = ['ob.and_operator', 'ob.or_operator']
K = ['ob.lequal', 'ob.grequal', 'ob.less', 'ob.greater']
L = ['ob.atleast', 'ob.atmost', 'ob.exactly']
M = [1, 2, 3]
productions = {"S": S, "A": A, "B": B, "C": C, "D": D, "E": E, "G": G, "H": H, "I": I, "J": J, "K": K, "L": L, "M": M}



def reproduce_rule_2(parse_tree, productions,start="S"):
    """ reproducing a pcfg rule based on a predetermined parse tree (could be a sub tree of another parse tree)
    NOTE: this function needs to be updated to be more concise """
    recode = parse_tree
    rule = start
    bound_vars = []
    # print(recode.keys())

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
    val_dict = {}


    # print(parse_tree)

    while any([i for i in ["S", "A"] if (i in [char for char in rule])]):
        srule = [char for char in rule]
        for i in range(0, len(srule)):
            if srule[i] == "S":
                if "S" not in list(recode.keys()):
                    print('s1')
                    return rule, bound_vars
                else:
                    if i_2 == len(recode["S"]):
                        print('s2')
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
                    print('a1')
                    return rule, bound_vars
                else:
                    if i_3 == len(recode["A"]):
                        print('a2')
                        return rule, bound_vars

                    else:
                        srule[i] = replacements[recode["A"][i_3]]
                        rule = "".join(srule)
                        i_3 += 1

    while any([i for i in ["B", "C", "D", "E", "G", "H", "I", "J", "K", "L", "M"] if (i in [char for char in rule])]):

        srule = [char for char in rule]
        for i in range(0, len(srule)):
            if srule[i] == 'B':
                print('hi')
                replacements = productions["B"]
                if "B" not in recode.keys():
                    print('b1')
                    return rule, bound_vars
                else:
                    if i_4 < len(recode["B"]):
                        # print([recode["B"]])
                        srule[i] = replacements[recode["B"][i_4]]
                        rule = "".join(srule)
                        i_4 += 1
                    elif i_4 >= len(recode["B"]):
                        print('cat')
                        return rule, bound_vars
                    # elif i_4 == len(recode["B"]):
                    #      i_4 += 1





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
                    print('c1')
                    return rule, bound_vars
                else:
                    if i_5 == len(recode["C"]):
                        print('c2')
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["C"][i_5]]
                        rule = "".join(srule)
                        i_5 += 1


            if srule[i] == 'D':
                # val_ind = 0
                if "D" not in recode.keys():
                    return rule, bound_vars
                else:
                    if i_6 == len(recode["D"]):
                        print('d2')
                        return rule, bound_vars
                    else:
                        feature = list(productions["D"].keys())[recode["D"][i_6]]
                nest_key = "D" + str(recode["D"][i_6])
                if nest_key not in list(val_dict.keys()):
                    val_dict[nest_key] = 0
                if nest_key not in recode.keys():
                    print('d3')
                    return rule, bound_vars
                else:
                    if i_6 == len(recode["D"]):
                        # print('d4')
                        return rule, bound_vars
                    else:
                        val_ind = 1
                        id_1 = recode["D" + str(recode["D"][i_6])]
                        value = productions["D"][feature][id_1[val_dict[nest_key]]]
                        replacement = str(value) + "," + "'" + feature + "'"
                        srule[i] = replacement
                        rule = "".join(srule)
                        i_6 += 1
                        val_dict[nest_key] +=1

                        # print('hijljljlj')


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
                    xxx=1

                    # return rule, bound_vars
                else:

                    # if i_11 > len(recode["J"]):
                    #     print('j1')
                    #     return rule, bound_vars
                    print(recode["J"])
                    if i_11 < len(recode["J"]):
                        srule[i] = replacements[recode["J"][i_11]]
                        rule = "".join(srule)
                        i_11 += 1
                    if i_4 == len(recode["B"]):
                        print('va')
                        return rule, bound_vars



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
                    print("l1")
                    return rule, bound_vars
                else:
                    if i_13 == len(recode["L"]):
                        print('l2')
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["L"][i_13]]
                        rule = "".join(srule)
                        i_13 += 1

            if srule[i] == 'M':
                replacements = productions["M"]
                if "M" not in recode.keys():
                    print("m1")
                    return rule, bound_vars
                else:
                    if i_14 == len(recode["M"]):
                        print("m2")
                        return rule, bound_vars
                    else:
                        srule[i] = str(replacements[recode["M"][i_14]])
                        rule = "".join(srule)
                        i_14 += 1
    print(val_dict)
    # reproduces a rule based on the parse tree and returns the rule as well as its bound variables
    # print('vaasa')
    return [rule, bound_vars]


#
def resample_rule(parse_tree):
    """ creating a random sub tree from a parse tree that is used as start for re-growing of a new parse tree """
    keys = [key for key in list(parse_tree.keys()) if key in ["S", "A", "B","C"]] # keeping only non-terminals
    key = rd.choice(keys)
    # key="S"
    ind = rd.choice(range(0, len(parse_tree[key])))  # selecting which production should be selected precisely
    # val = parse_tree[key][ind]       # selecting feature/value of a production
    sub_tree = {i: parse_tree[i][:1] for i in parse_tree if  # creating sub tree
                list(parse_tree.keys()).index(i) <= list(parse_tree.keys()).index(key)}
    sub_tree[key] = sub_tree[key][:ind]
    # print(sub_tree)
    # if key is B, its important to take J and C as well as this determines which booleans were already used
    if key == "B":
        js = sum([1 for i in parse_tree["B"][:ind] if i ==1])-1
        cs = sum([1 for i in parse_tree["B"][:ind] if i ==0])-1
        if js >= 1:
            sub_tree["J"] = parse_tree["J"][:1]
        # if cs >= 1:
        #     sub_tree["C"] = parse_tree["C"][:1]
    # if key is C, its important to consider B as well
    if key == "C":
        bs = [i for i in range(len(parse_tree["B"])) if parse_tree["B"][i] in parse_tree[key]]
        sub_tree["B"] = parse_tree["B"][:bs[len(parse_tree[key][:ind])]]
        js = sum([1 for i in sub_tree["B"] if i ==1]) - 1
        if js >= 1:
            sub_tree["J"] = parse_tree["J"][:js]
    sub_tree = {k:v for k,v in sub_tree.items() if v}
    # print(sub_tree)
    return sub_tree


def reproduce_rule(parse_tree, productions):
    """ reproducing a pcfg rule based on a predetermined parse tree (could be a sub tree of another parse tree)
    NOTE: this function needs to be updated to be more concise """
    recode = parse_tree
    rule = "S"
    bound_vars = []
    # print(recode)
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
                    # print('s1')
                    return rule, bound_vars
                else:
                    if i_2 == len(recode["S"]):
                        # print('s2')
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
                    # print('a1')
                    return rule, bound_vars
                else:
                    if i_3 == len(recode["A"]):
                        # print('a2')
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
                    # print('b1')
                    return rule, bound_vars

                else:

                    if i_4 == len(recode["B"]):
                        # print('b2')
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
                    # print('c1')
                    return rule, bound_vars
                else:
                    if i_5 < len(recode["C"]):
                        srule[i] = replacements[recode["C"][i_5]]
                        rule = "".join(srule)
                        i_5 += 1

                    else:
                        if i_4 > len(recode["B"]):
                            # print('j2')
                            return rule, bound_vars


                        # print('c2')
                        # return rule, bound_vars


            if srule[i] == 'D':
                if "D" not in recode.keys():
                    if i_4 > len(recode["B"]):
                        # print('d1')
                        return rule, bound_vars
                    if "J" not in recode.keys():
                            if i_4 == len(recode["B"]):
                                return rule, bound_vars
                    if i_11 == len(recode["J"]):
                        if i_4 == len(recode["B"]):
                            return rule, bound_vars
                else:
                    if i_6 == len(recode["D"]):
                        # print('d2')
                        return rule, bound_vars
                    else:
                        feature = list(productions["D"].keys())[recode["D"][i_6]]
                    nest_key = "D" + str(recode["D"][i_6])
                    if nest_key not in recode.keys():
                        # print('d3')
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
                    # print('j1')
                    return rule, bound_vars
                else:
                    if i_11 < len(recode["J"]):
                        srule[i] = replacements[recode["J"][i_11]]
                        rule = "".join(srule)
                        i_11 += 1
                        if i_4 > len(recode["B"]):
                            # print('j1')
                            return rule, bound_vars
                    #
                        # print('j2')
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["J"][i_11]]
                        rule = "".join(srule)
                        i_11 += 1
                        if i_4 > len(recode["B"]):
                            # print('j3')
                            return rule, bound_vars

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
                    # print('l1')
                    return rule, bound_vars
                else:
                    if i_13 == len(recode["L"]):
                        # print('l3')
                        return rule, bound_vars
                    else:
                        srule[i] = replacements[recode["L"][i_13]]
                        rule = "".join(srule)
                        i_13 += 1

            if srule[i] == 'M':
                replacements = productions["M"]
                if "M" not in recode.keys():
                    # print('m1')
                    return rule, bound_vars
                else:
                    if i_14 == len(recode["M"]):
                        # print('m2')
                        return rule, bound_vars
                    else:
                        srule[i] = str(replacements[recode["M"][i_14]])
                        rule = "".join(srule)
        # return [rule, bound_vars]
        # print(rule)
    # reproduces a rule based on the parse tree and returns the rule as well as its bound variables
    return [rule, bound_vars]



def resample_rule_2(parse_tree, productions):
    """ this does some surgery to the tree and ensures that some part is changed while the rest remains the same.
    # it does, however, not guarantee that the result is a valid expression """
    d2 = None
    a = parse_tree["B"]
    print('parsetree')
    print(parse_tree)
    keys = [key for key in list(parse_tree.keys()) if key in ["S", "D", "B", "J", "L","D0", "D1"]] # keeping only non-terminals
    key = rd.choice(keys)             # selecting random production from parse tree
    key = "S"
    print(key)
    ind = rd.choice(list(range(0, len(parse_tree[key]))))
    # special case for D since these are features and they need to create another entry for values
    if key == 'D':
        feats =  {"colour": ["'red'", "'blue'", "'green'"],
                  "size": [1, 2, 3],
                  "orientation": ["'upright'", "'lhs'", "'rhs'", "'strange'"],
                  "grounded": ["'no'", "'yes'"]}
        # sample new feature key
        key = rd.choice(list(feats.keys()))

        # pick a colour value
        ind2 = productions['D'][key].index(rd.choice(productions['D'][key]))   # index of feature key
        # get colour

        ind = list(productions['D'].keys()).index(key)
        print(ind)
        #pick place in parse where colour should be inserted
        replace = rd.randint(0,len(parse_tree['D'])-1)
        init_v = parse_tree['D'][replace]
        parse_tree['D'][replace] = ind

        # create value index
        d2 = 'D' + str(ind)
        if d2 in parse_tree:
            # if value already in parse pick random value from potential lists of values that should be replaced
            rd_val = rd.choice(list(range(0,len(parse_tree[d2]))))
            # old = parse_tree[d2][rd_val]
            # get present value index that should be replaced
            ps_v = parse_tree[d2][rd_val]
            while ps_v == [ind2]: # if value index is the same as the one sampled, sample again
                ind2 = productions['D'][key].index(rd.choice(productions['D'][key]))


            if len(parse_tree[d2]) == 1 or rd_val == (len(parse_tree[d2]) -1):
                # if parse tree has just one value for the present feature, append the new feature

                if init_v != parse_tree["D"][replace]:
                    parse_tree[d2].append(ind2)
                else:
                    parse_tree[d2][sum([1 for f in parse_tree["D"] if f == str(ind)])-1] = ind2
                # print('car')
            else:
                # print('meowmewo')
                old = parse_tree[d2]
                new = [ind2]
                if init_v != parse_tree["D"][replace]:
                    parse_tree[d2] = old[:rd_val] + new + old[rd_val:]
                else:
                    parse_tree[d2][rd_val] = ind2
        else:
            parse_tree[d2] = [ind2]

    # rest of the mtfck keys

    if key == "S":
        val = rd.choice(productions[key])
        ind = productions[key].index(val)
        while [ind] == parse_tree[key]:
            val = rd.choice(productions[key])
            ind = productions[key].index(val)
        if parse_tree["S"] == [2]:
            flip = rd.choice([0,1])
            if flip == 0:
                del(parse_tree["M"], parse_tree["L"])
            else:
                indl = rd.choice([f for f in [0,1,2] if f != parse_tree["L"][0]])
                ind = 2
                parse_tree["L"] = [indl]
        if val == "L(lambda xN: A, M, X)":
            parse_tree["M"] = [0]
            parse_tree["L"] = [productions["L"].index(rd.choice(productions["L"]))]
        parse_tree[key] = [ind]

    # B
    if key == "B":
        val = rd.choice(["ob.not_operator(B)", "J(B,B)"])
        val = "J(B,B)"
        if val == "ob.not_operator(B)":
            indb = rd.choice(list(range(0, len(parse_tree["B"]))))
            init_val = parse_tree["B"][indb]
            if init_val == 2:
                del(parse_tree["B"][indb])
            elif init_val != 2:
                parse_tree["B"].insert(indb,2)
        elif val == "J(B,B)":
            if 1 in a:
                parse_tree["B"] = a
                indj = rd.choice(list(range(0, len(parse_tree["J"]))))

                if parse_tree["J"][indj] == 0:
                    parse_tree["J"][indj] = 1
                elif parse_tree["J"][indj] == 1:
                    parse_tree["J"][indj] = 0
            else:
                indb = rd.choice(list(range(0, len(parse_tree["B"]))))
                parse_tree["J"] = [rd.choice([0,1])]
                parse_tree["B"] = a[:indb] + [1] + a[indb:] + [0]
                parse_tree["C"] = parse_tree["C"] + [0]
                current_feat = [k[1] for k in list(parse_tree.keys()) if len(k) == 2][0]
                new_feat = rd.choice([f for f in [0,1,5,6] if f != current_feat])
                print(new_feat)
                feat_ind = "D" + str(new_feat)
                newkey = list(productions["D"].keys())[new_feat]
                new_val_ind = rd.choice(list(range(0, len(productions["D"][newkey]))))
                parse_tree["D"] = [new_feat] + parse_tree["D"]
                if feat_ind in list(parse_tree.keys()):
                    parse_tree[feat_ind].append(new_val_ind)
                else:
                    parse_tree[feat_ind] = [new_val_ind]

    if key == "J":
        indj = rd.choice(list(range(0,len(parse_tree["J"]))))
        if parse_tree["J"][indj] == 0:
            parse_tree["J"][indj] = 1
        elif parse_tree["J"][indj]  == 1:
            parse_tree["J"][indj] = 0

    if key == "L":
        options = [l for l in [0,1,2] if l != parse_tree["L"][0]]
        parse_tree["L"] = [rd.choice(options)]


    #kicking out all redundant keys
    list_val_keys = []
    for val_ind in parse_tree["D"]:
        val_key = "D"+str(val_ind)
        sum_val_occurence = sum([1 for v in parse_tree["D"] if v == val_ind])
        # if val_key in parse_tree.keys():
        parse_tree[val_key] = parse_tree[val_key][:sum_val_occurence+1]
        list_val_keys.append(val_key)
    parse_tree = {key:val for key, val in parse_tree.items() if len(key) == 1 or key in list_val_keys or key == d2}

    return parse_tree


for i in range(1):
    rule = "ob.forall(lambda x1: ob.not_operator(ob.equal(x1, 3,'size')), X)"
    sub_tree = resample_rule_2({'S': [1], 'A': [0], 'B': [2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 0, 0, 0,0,0,0], 'J': [0, 0, 0, 1,1], 'C': [0, 0, 0,0,0,0], 'D': [6,6,6,6,6,6], 'D6': [1,1, 0,0,0,0]},productions)
    print(sub_tree)
    print(reproduce_rule_2(sub_tree, productions))
    # print(reproduce_rule_2({'S': [2], 'A': [0], 'B': [2, 1, 0, 0], 'C': [0, 0], 'D': [6, 0], 'D0': [0], 'J': [1], 'D6': [1], 'M': [0], 'L': [1]}, productions))
    # sub_t = reproduce_rule_2(sub_tree, productions)
    # print(sub_tree)
    # rule looks different - change feature value but not feature, both remove,
    # do not allow feature to be changed - maybe rare costly edits
    #value is still a fundamental change
    # certain edits are very natural, like number,
    # S most common, addition of or / and as second feature. Productions are more immutable
    # S,A,B,C,D,...
