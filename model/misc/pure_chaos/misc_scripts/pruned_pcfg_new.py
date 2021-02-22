"""
Created on Sun Mar 15 15:01:21 2020

@author: jan-philippfranken
"""
###############################################################################
############################### PCFG GENERATOR ################################
###############################################################################


####################### General imports #######################################
import numpy as np
import pandas as pd
import re


####################### Custom imports ########################################
from rules import rules


class pcfg_generator(rules):
    def __init__self(self):
        self.__init__self(self)

    def probs_list(self, probs):
        """ translates the full parse tree of a rule into a list of lists that is used to compute the prior of
        each rule (see formal learning model for details on the computation of the prior) """
        prod_l = []
        for key in probs.keys():
            if isinstance(probs[key], list):   # currently able to translate parse trees with nested trees (i.e. dictionary within dictionary)
                prod_l.append(probs[key])
            elif isinstance(probs[key], dict):
                keys_nested = probs[key].keys()
                cover_list = []
                for key_nested in keys_nested:
                    cover_list.append(np.sum(probs[key][key_nested]))
                    prod_l.append(probs[key][key_nested])
                prod_l.append(cover_list)
        return prod_l

    def pd_options(self, pds):
        """ returns an empty parse tree including all possible non-terminals in productions """
        keys = pds.keys()
        pd_opts = {}
        for i in keys:
            if isinstance(pds[i], list):   # also allows for nested dictionary construction
                pd_opt = [0] * len(pds[i])
                pd_opts[i] = pd_opt
            elif isinstance(pds[i], dict):
                keys_nested = pds[i].keys()
                pd_opts_nested = {}
                for i_2 in keys_nested:
                    pd_opt_nested = [0] * len(pds[i][i_2])
                    pd_opts_nested[i_2] = pd_opt_nested
                pd_opts[i] = pd_opts_nested
        return pd_opts

    def generate_res(self,
                 productions,
                 rule,
                 bound_vars,
                 prec,
                 feat_probs=[[0.333333333,.333333333,.333333333],[0.333333333,.333333333,.333333333],[0],[0],[0],[.25,.25,.25,.25],[.5,.5]],
                 Swin =  [0.333333333,.333333333,.333333333],
                 Awin =  [.5,.5],
                 Bwin = [0.333333333,.333333333,.333333333],
                 Cwin = [.2,.2,.2,.2,.2],
                 Cwin2 = [.2,.2,.2,.2,.2],
                 Dwin =  [.25,.25,0,0,0,.25,.25],
                 Ewin = [.25,.25,.25,.25],
                 Gwin = [.25,.25,0,0,0,.25,.25],
                 Hwin =  [.25,.25,.25,.25],
                 Iwin = [],
                 Jwin =  [.5,.5],
                 Kwin =  [0],
                 Lwin =  [0.333333333,.333333333,.333333333],
                 Mwin = [1,0,0]):
        """ generates a rule according to the specified production, the start rule (rule argument), the start bound
        variables (bound_vars argument), and the start frame storing rule details (prec) """
        prec = prec
        rule = rule
        bound_vars = bound_vars
        probs = self.pd_options(productions)
        probs_ordered = {}
        # print(feat_probs)
        # print(Dwin)
        # print('hi')
        # print(Cwin)

        # NOTE: this generation procedure is currently specificly focussing on the productions specified in Bramley et al. (2018)
        # this needs to be updated for other languages of thought / production rules to be more flexible
        while any([i for i in ["S", "A"] if(i in [char for char in rule])]):
            srule = [char for char in rule]
            for i in range(0, len(srule)):

                if srule[i] == 'S':
                    bound_vars.append(["x" + str(i) for i in range(1, len(bound_vars) + 1)])
                    Sk = [re.sub("N", str(len(bound_vars)), i) for i in productions["S"]]
                    # Sw = np.repeat(1/len(Sk), len(Sk))
                    # if Swin != False:
                    Sw = Swin
                    ix = np.random.choice(np.arange(0, len(productions["S"])), 1, p = Sw)
                    replacement = Sk[int(ix)]
                    prec = prec.append(pd.DataFrame({"from": "S", "to": productions["S"][int(ix)], "toix": ix, "li": Sw[int(ix)]}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    probs["S"][int(ix)] += 1
                    if "S" in probs_ordered:
                        probs_ordered["S"].append(int(ix))
                    else:
                        probs_ordered["S"] = []
                        probs_ordered["S"].append(int(ix))

                if srule[i] == "A":
                    # Aw = np.repeat(1/len(productions["A"]), len(productions["A"]))
                    # if Awin != False:
                    Aw = Awin
                    if len(bound_vars) >= 3:
                        # print(">3 variable bindings ")
                        Aw = [1,0]
                    # print(Aw)
                    ix = np.random.choice(np.arange(0, len(productions["A"])), 1, p = Aw)
                    replacement = productions["A"][int(ix)]
                    prec = prec.append(pd.DataFrame({"from": "A", "to": replacement, "toix": ix, "li": Aw[int(ix)]}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    probs["A"][int(ix)] += 1
                    if "A" in probs_ordered:
                        probs_ordered["A"].append(int(ix))
                    else:
                        probs_ordered["A"] = []
                        probs_ordered["A"].append(int(ix))

        while any([i for i in ["B", "C", "D", "E", "G", "H", "I", "J", "K", "L", "M"] if(i in [char for char in rule])]):
            srule = [char for char in rule]
            for i in range(0, len(srule)):
                """ for B"""
                if srule[i] == 'B':
                    # Bw = np.repeat(1/len(productions["B"]), len(productions["B"]))
                    # if Bwin != False:
                    Bw = Bwin
                    if np.sum(prec['from'] == 'B') > 10:
                        # print('>10 B expansions')
                        Bw = [1,0,0]
                    ix = np.random.choice(np.arange(0, len(productions["B"])), 1, p = Bw)
                    replacement = productions["B"][int(ix)]
                    prec = prec.append(pd.DataFrame({"from": "B", "to": replacement, "toix": ix, "li": Bw[int(ix)]}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    probs["B"][int(ix)] += 1
                    if "B" in probs_ordered:
                        probs_ordered["B"].append(int(ix))
                    else:
                        probs_ordered["B"] = []
                        probs_ordered["B"].append(int(ix))


                if srule[i] == 'C':
                    Cw = Cwin

                    # np.repeat(1/len(productions["C"]), len(productions["C"]))
                    if len(bound_vars) == 1:
                        Ck = [re.sub("N", "1", i) for i in productions["C"]]
                        Cw[2:6] = [0 for i in Cw[2:6]]
                        Cw = (Cw/np.sum(Cw))

                    else:
                        Cw = Cwin2
                        # print(Cw)
                        reps = np.random.choice(np.arange(0, len(bound_vars)), 2)
                        Ck = [re.sub("N", str(reps[0] + 1), i) for i in productions["C"]]
                        Ck = [re.sub("O", str(reps[1] + 1), i) for i in Ck]

                    # print(Cw)
                    ix = np.random.choice(np.arange(0, len(Ck)), 1, p = Cw)
                    replacement = Ck[int(ix)]
                    prec = prec.append(pd.DataFrame({"from": "C", "to": Ck[int(ix)], "toix": ix, "li": Cw[int(ix)]}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    probs["C"][int(ix)] += 1
                    if "C" in probs_ordered:
                        probs_ordered["C"].append(int(ix))
                    else:
                        probs_ordered["C"] = []
                        probs_ordered["C"].append(int(ix))

                if srule[i] == 'D':
                    # Dw =  np.repeat(1/len(productions["D"]), len(productions["D"]))
                    # if Dwin != False:
                    Dw = Dwin
                    ix = np.random.choice(np.arange(0, len(productions["D"])), 1, p = Dw)
                    feature = list(productions["D"].keys())[int(ix)]
                    feat_probs_trial = feat_probs[int(ix)]
                    # print(feat_probs_trial)
                    # print(feat_probs)
                    # print(feat_probs_trial)
                    # print(len(productions["D"][feature]))
                    vix = np.random.choice(np.arange(0, len(productions["D"][feature])), 1, p=feat_probs_trial)
                    value = productions["D"][feature][int(vix)]

                    # print(feature)
                    # print(value)
                    # print(Dw)
                    # print(feat_probs_trial)
                    # print('vat')
                    replacement = str(value) + "," + "'" + feature + "'"
                    prec = prec.append(pd.DataFrame({"from": "Ef", "to": feature, "toix": ix, "li": Dw[int(ix)]}), ignore_index=True)
                    prec = prec.append(pd.DataFrame({"from": "Ev", "to": value, "toix": vix, "li": 1/len(productions["D"][feature])}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    # probs["D"][int(ix)] += 1
                    probs["D"][feature][int(vix)] += 1
                    if "D" in probs_ordered:
                        probs_ordered["D"].append(int(ix))
                    else:
                        probs_ordered["D"] = []
                        probs_ordered["D"].append(int(ix))
                    if "D" + (str(ix[0])) in probs_ordered:
                        probs_ordered["D" + (str(ix[0]))].append(int(vix))
                    else:
                        probs_ordered["D" + (str(ix[0]))] = []
                        probs_ordered["D" + (str(ix[0]))].append(int(vix))



                    # print(Dw)
                if srule[i] == 'E':
                    # Ew = np.repeat(1/len(productions["E"]), len(productions["E"]))
                    # if Ewin != False:
                    Ew = Ewin
                    ix = np.random.choice(np.arange(0, len(productions["E"])), 1, p = Ew)
                    feature = list(productions["E"].keys())[int(ix)]
                    # vix = np.random.choice(np.arange(0, len(productions["E"][feature])), 1)
                    # value = productions["E"][feature][int(vix)]
                    # print(feature)
                    vix = np.random.choice(np.arange(0, len(productions["E"][feature])), 1, p=feat_probs[1])

                    value = productions["E"][feature][int(vix)]
                    # print(feat_probs[1])
                    replacement = str(value) + "," + "'" + feature + "'"
                    prec = prec.append(pd.DataFrame({"from": "Ef", "to": feature, "toix": ix, "li": Ew[int(ix)]}), ignore_index=True)
                    prec = prec.append(pd.DataFrame({"from": "Ev", "to": value, "toix": vix, "li": 1/len(productions["E"][feature])}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    # probs["E"][int(ix)] += 1
                    probs["E"][feature][int(vix)] += 1
                    if "E" in probs_ordered:
                        probs_ordered["E"].append(int(ix))
                    else:
                        probs_ordered["E"] = []
                        probs_ordered["E"].append(int(ix))
                    if "E" + (str(ix[0])) in probs_ordered:
                        probs_ordered["E" + (str(ix[0]))].append(int(vix))
                    else:
                        probs_ordered["E" + (str(ix[0]))] = []
                        probs_ordered["E" + (str(ix[0]))].append(int(vix))


                if srule[i] == 'G':
                    # Gw = np.repeat(1/len(productions["G"]), len(productions["G"]))
                    # if Gwin != False:
                    Gw = Gwin
                    ix = np.random.choice(np.arange(0, len(productions["G"])), 1, p = Gw)
                    replacement = productions["G"][int(ix)]
                    prec = prec.append(pd.DataFrame({"from": "G", "to": productions["G"][int(ix)], "toix": ix, "li": Gw[int(ix)]}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    probs["G"][int(ix)] += 1
                    if "G" in probs_ordered:
                        probs_ordered["G"].append(int(ix))
                    else:
                        probs_ordered["G"] = []
                        probs_ordered["G"].append(int(ix))

                if srule[i] == 'H':
                    # Hw = np.repeat(1/len(productions["H"]), len(productions["H"]))
                    # if Hwin != False:
                    Hw = Hwin
                    ix = np.random.choice(np.arange(0, len(productions["H"])), 1, p = Hw)
                    replacement = productions["H"][int(ix)]
                    prec = prec.append(pd.DataFrame({"from": "H", "to": productions["H"][int(ix)], "toix": ix, "li": Hw[int(ix)]}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    probs["H"][int(ix)] += 1
                    if "H" in probs_ordered:
                        probs_ordered["H"].append(int(ix))
                    else:
                        probs_ordered["H"] = []
                        probs_ordered["H"].append(int(ix))

                if srule[i] == 'I':
                    # Iw = np.repeat(1/len(productions["I"]), len(productions["I"]))
                    # if Iwin != False:
                    Iw = Iwin

                    # if len(productions["I"]) > 1:
                    #     ix = np.random.choice(np.arange(0, len(productions["I"])), 1, p = Iw)
                    # else:
                    ix = 0
                    replacement = productions["I"][ix]
                    prec = prec.append(pd.DataFrame({"from": "I", "to": productions["I"], "toix": ix, "li": Iw[int(ix)]}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    probs["I"][int(ix)] += 1
                    if "I" in probs_ordered:
                        probs_ordered["I"].append(int(ix))
                    else:
                        probs_ordered["I"] = []
                        probs_ordered["I"].append(int(ix))

                if srule[i] == 'J':
                    # Jw = np.repeat(1/len(productions["J"]), len(productions["J"]))
                    # if Jwin != False:
                    Jw = Jwin
                    if len(productions["J"]) > 1:
                        ix = np.random.choice(np.arange(0, len(productions["J"])), 1, p = Jw)
                    else:
                        ix = 0

                    replacement = productions["J"][int(ix)]
                    prec = prec.append(pd.DataFrame({"from": "J", "to": productions["J"][int(ix)], "toix": ix, "li": Jw[int(ix)]}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    probs["J"][int(ix)] += 1
                    if "J" in probs_ordered:
                        probs_ordered["J"].append(int(ix))
                    else:
                        probs_ordered["J"] = []
                        probs_ordered["J"].append(int(ix))

                if srule[i] == 'K':
                    # Kw = np.repeat(1/len(productions["K"]), len(productions["K"]))
                    # if Kwin != False:
                    Kw = Kwin
                    # if len(productions["K"]) > 1:
                    ix = np.random.choice(np.arange(0, len(productions["K"])), 1, p = Kw)
                    # else:
                    #     ix = 0
                    replacement = productions["K"][int(ix)]
                    prec = prec.append(pd.DataFrame({"from": "K", "to": productions["K"][int(ix)], "toix": ix, "li": Kw[int(ix)]}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    probs["K"][int(ix)] += 1
                    if "K" in probs_ordered:
                        probs_ordered["K"].append(int(ix))
                    else:
                        probs_ordered["K"] = []
                        probs_ordered["K"].append(int(ix))

                if srule[i] == 'L':
                    # Lw = np.repeat(1/len(productions["L"]), len(productions["L"]))
                    # if Lwin != False:
                    Lw = Lwin
                    # if len(productions["L"]) > 1:
                    ix = np.random.choice(np.arange(0, len(productions["L"])), 1, p = Lw)
                    # else:
                    #     ix = 0

                    replacement = productions["L"][int(ix)]
                    prec = prec.append(pd.DataFrame({"from": "L", "to": productions["L"][int(ix)], "toix": ix, "li": Lw[int(ix)]}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    probs["L"][int(ix)] += 1
                    if "L" in probs_ordered:
                        probs_ordered["L"].append(int(ix))
                    else:
                        probs_ordered["L"] = []
                        probs_ordered["L"].append(int(ix))

                if srule[i] == 'M':
                    # Mw = np.repeat(1/len(productions["M"]), len(productions["M"]))
                    # if Mwin != False:
                    Mw = Mwin
                    # if len(productions["M"]) > 1:
                    ix = np.random.choice(np.arange(0, len(productions["M"])), 1, p = Mw)
                    # else:
                    #     ix = 0
                    replacement = str(productions["M"][int(ix)])
                    prec = prec.append(pd.DataFrame({"from": "M", "to": productions["M"][int(ix)], "toix": ix, "li": Mw[int(ix)]}), ignore_index=True)
                    srule[i] = replacement
                    rule = "".join(srule)
                    probs["M"][int(ix)] += 1

                    if "M" in probs_ordered:
                        probs_ordered["M"].append(int(ix))
                    else:
                        probs_ordered["M"] = []
                        probs_ordered["M"].append(int(ix))

        # rule = a string entailing the generated rule, prec: df containing information about rule details (e.g., number of productions)
        # probs: full parse tree / productions dictionary counting number each production has been used
        # prob_l: list version of probs, used to calculate the prior of a rule (see formal learning model)
        # prod_d: short version of probs, used during the regeneration of rules
        # bv list including lists for all bound variables
        # print(rule)
        # print('whale')
        return {"rule": rule, "prec": prec, "probs": probs, "prod_l": self.probs_list(probs), "prod_d": probs_ordered, "bv": bound_vars}

###############################################################################
###############################################################################


