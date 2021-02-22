####################### General imports #######################################
import numpy as np
import pandas as pd

####################### Custom imports ########################################
from pcfg import pcfg_generator

ob = pcfg_generator()       # instantiating pcfg_generator

def sample_rules(productions, data, ground_truth, start="S",
                 start_frame=pd.DataFrame({"from": [], "to": [], "toix": [], "li": []}), iter=100):
    """ generating n=iter pcfg based rules that are evaluated based on a deterministic likelihood (i.e. match data vs. do no match
    data) """

    df_rules = pd.DataFrame({"rulestring": [""] * iter,  # data frame for storing rules and other relevant properties
                             "productions": [None] * iter,
                             "prod_l": [None] * iter,
                             "li": [None] * iter,
                             "n_pds": [None] * iter,
                             "results": [None] * iter})
    rule_details = {}  # additional rule details returned in the end

    for i in range(0, iter):  # generating n rules based on iter value
        t = ob.generate(productions, start, [], start_frame)  # appending properties to the data frame
        print(t["bv"])
        df_rules["rulestring"][i] = t["rule"]
        df_rules["li"][i] = np.prod(t["prec"]["li"])
        df_rules["productions"][i] = t["prod_d"]
        df_rules["n_pds"][i] = len(t["prec"])
        df_rules["prod_l"][i] = t["prod_l"]
        rule_details[str(i)] = t["prec"]

        global X  # evaluating rule (requires global X)
        res = []  # results for each rule and scenes
        for i_2 in range(0, len(data)):
            X = data[i_2]
            res.append(eval(t["rule"]))
        z = [a and b for a, b in zip(ground_truth, res)]  # comparing results with ground truth
        df_rules["results"][i] = z

    return df_rules
