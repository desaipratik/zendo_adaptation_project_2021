"""
Created on Tue Aug 12 15:56:00 2020

@author: jan-philippfranken
"""
###############################################################################
########################### Tree Regrower  ###################################
###############################################################################

####################### General imports #######################################
import random as rd
import copy
import pandas as pd

####################### Custom imports ########################################
from recode_rule_to_list import rule_translator


class tree_regrower(rule_translator):
    '''this fellow does some regrowing to a given tree'''
    def __init__self(self):
        self.__init__self(self)

    def regrow_tree(self, t, prod, replacements,non_terminal_list =  ['S','A','C','B']):
        '''this function regrows a tree based on an input tree and returns sufficient information for growing new sub tree'''
        # first getting all information from initial tree
        productions = copy.deepcopy(prod)
        t_rule = t['rule']         # initial rule
        t_prec = t['prec']
        # t_prec = pd.read_csv('test.csv')
        t_prod = t_prec['from']
        t_bv = t['bv']             # initial bound variables
        # t_bv = ['x1']
        t_prime_bv = t_bv.copy()   # new bound variables (might be changed later)
        t_list = self.string_to_list(t_rule)  # transforming rule into list of list
        # t_list = ['Z.forall', ['lambda', 'x1', ':', 'Z.not_operator', ['Z.not_operator', ['Z.or_operator', ['Z.and_operator', ['Z.or_operator', ['Z.not_operator', ['Z.not_operator', ['Z.equal', ['x1', 'green', 'colour']]], 'Z.not_operator', ['Z.equal', ['x1', 'green', 'colour']]], 'Z.equal', ['x1', 'blue', 'colour']], 'Z.not_operator', ['Z.equal', ['x1', 'blue', 'colour']]]]], 'X']]
        ind_nested = self.get_inds(t_list)    # gettinng elements and all nested indexes
        # print(ind_nested)
        # print(t_list)

        # then sampling new node and replacements
        t_prod_inds = list(range(0, len(t_prod)))
        nt_inds = [index for index,prod in zip(t_prod_inds,t_prod) if prod in non_terminal_list]
        nt_ind = rd.choice(nt_inds)     # selecting random nonterminal index from which tree will be regrown
        # nt_ind = 5/
        # nt_in/d = 4
        nt = t_prod[nt_ind]

#         # print(nt_ind)
#         # print('indaabove')

        if len(t_bv) == 1:
            productions[nt] = ['Z.equal(x1, D)', 'K(x1, E)']


        p_ind = rd.choice(list(range(0, len(productions[nt]))))
        # p_ind = 0
#         # print(p_ind)
#         # print('pindabove')
        new_p = productions[nt][p_ind]
        new_prod_prob = 1/len(productions[nt])
        cut = 0  # default variable that might change to 1 if nt is "S" (since then more needs to be removed from t_prec, see below)

        # now checking which nt was selected and reacting accordinngly
        if nt == "B" or nt == "C":
            b_inds = [index for index,prod in zip(t_prod_inds,t_prod) if prod in [nt]] # getting indexes for B only
            n_b_select = b_inds.index(nt_ind)     # getting the exact B (ie which one from many has been chosen if many are available)
            # now use the above to get the index of the selected be from the lists including all features
            all_inds = list(range(0, len(ind_nested)))
            b_inds = [index for index,prod in zip(all_inds,ind_nested) if prod in replacements[nt]]
            spec_ind = ind_nested[b_inds[n_b_select]+1]
            spec_ind_plus = spec_ind.copy()
            spec_ind_plus[len(spec_ind_plus)-1] += 1


            # ensures that no nonsense comparisons can be made
            bound_vars = list(range(1, len(t_prime_bv)+1))
            for ch in ["N", "O"]:
                if ch in new_p:
                    rand_ind = bound_vars.index(rd.choice(bound_vars))
                    new_p = new_p.replace(ch, str(bound_vars[rand_ind]))
                    if len(t_bv) >= 2:
                        del(bound_vars[rand_ind])


            # now transforming the indeces for the deeplist into strings to then replace items in the deeplist
            spec_ind_l = [str([ind]) for ind in spec_ind]
            spec_ind_plus_l = [str([ind]) for ind in spec_ind_plus]
            spec_ind_s = ''.join(spec_ind_l)
            spec_ind_plus_s = ''.join(spec_ind_plus_l)
            t_prime_list = copy.deepcopy(t_list)

            # replacing part of list with new part and removing the part immediately following the initial part
            exec('t_prime_list' + spec_ind_s + '=' + 'new_p')
            del_in_prec = eval('t_prime_list' + spec_ind_plus_s)  # quickly getinng the items that need to be deleted
            exec('del(t_prime_list' + spec_ind_plus_s + ')')

            # finally we need to get t_prime rule and create t_prime prec
            t_prec_to = list(t_prec['to'])
            t_prec_from = list(t_prec['from'])
            t_prime_rule = self.list_to_string(t_prime_list)
            t_prime_prec = t_prec
            t_prime_prec.at[nt_ind, 'from'] = nt
            t_prime_prec.at[nt_ind, 'to'] = new_p
            t_prime_prec.at[nt_ind, 'toix'] = float(p_ind)
            t_prime_prec.at[nt_ind, 'li'] = new_prod_prob
#             # print(t_prime_prec['to'])
            # some items from t_prime_prec need to be deleted (those that were only used in the initial t_prec); might be nested list
            del_in_prec = self.get_list(del_in_prec)
            del_in_prec_2 = [str(i) if isinstance(i, int) else "'" + i + "'" if isinstance(i, str) and '.' not in i else i for i in del_in_prec]
            del_in_prec = del_in_prec + del_in_prec_2
            prec_inds = list(range(0, len(t_prec['to'])))

            # need to apply some operations to t_prec_to first
            to_ind = 0
            for to in t_prec_to:
                for ch in ["("]:
                    if isinstance(to, str):
                        if ch in to:
                            cut = to.index(ch)
                            to = to[:cut]
                to_ind += 1

            # now we can identify which elements need ot be dropped
            drop_inds = [i for i,to in zip(prec_inds,t_prec_to) if to in del_in_prec and i > nt_ind]
            drop_inds_new = []
            for i in drop_inds:
                if t_prec_from[i] != "M":
                    drop_inds_new.append(i)

#             # print(t_prec_to)
            # if isinstance(t_prime_prec['to'][nt_ind +1], str):
            #     if t_prime_prec['to'][nt_ind +1][0] == "Z":
            #         if t_prime_prec['to'][nt_ind +2] not in productions["G"] or t_prime_prec['to'][nt_ind +2] not in productions["I"]:
            #             drop_inds.append(nt_ind+1)

            drop_inds_new = list(set(drop_inds_new)) # getting rid of duplicates
#             # print(drop_inds_new)

            if nt == "C" or "B":
                t_prime_prec = t_prime_prec.drop(drop_inds_new)

            # elif nt == "B":
            #     t_prime_prec = t_prime_prec.loc[:nt_ind:]


            return {"t_prime_rule": t_prime_rule, "t_prime_bv": t_prime_bv, "t_prime_prec": t_prime_prec, "nt_ind": nt_ind, "nt": nt}


        # this is a bit hacky since it does something to the number of bound variables
        elif nt == 'A':
            t_prime_list = t_list
            if nt_ind == 5:
                for ch in ["S"]:
                    if ch in new_p:
                        new_p = new_p.replace(ch, "B")
                t_prime_list[1][4][4][3] = new_p
                del(t_prime_list[1][4][4][4])
            elif nt_ind == 3:
                t_prime_bv = t_prime_bv[:2]
                t_prime_list[1][4][3] = new_p
                del(t_prime_list[1][4][4])
            elif nt_ind == 1:
                t_prime_bv = t_prime_bv[:1]
                t_prime_list[1][3] = new_p
# #                 # print('hi')
                del(t_prime_list[1][4])

        # also for S but simpler
        elif nt == 'S':
            # cut=1
            t_prime_list = t_list
            sum_s = sum([1 for a in t_prod if a in ["S"]])
            if nt_ind == 4:
                t_prime_bv = t_prime_bv[:3]
                for ch in ["A"]:
                    if ch in new_p:
                        new_p = new_p.replace(ch, "B")
                for ch in ["N"]:
                    if ch in new_p:
                        new_p = new_p.replace(ch, "3")
                t_prime_list[1][4][3] = new_p
                del(t_prime_list[1][4][4])
            elif nt_ind == 2:
                t_prime_bv = t_prime_bv[:2]
                for ch in ["N"]:
                    if ch in new_p:
                        new_p = new_p.replace(ch, "2")
                t_prime_list[1][3] = new_p
                del(t_prime_list[1][4])
            elif nt_ind == 0:
                t_prime_prec = pd.DataFrame({"from": [], "to": [], "toix": [], "li": []})
                t_prime_rule = "S"
                t_prime_bv = []
                t_prime_list[0] = "S"
                del(t_prime_list[1])
                return {"t_prime_rule": t_prime_rule, "t_prime_bv": t_prime_bv, "t_prime_prec": t_prime_prec, "nt_ind": nt_ind, "nt": nt}


        t_prime_rule = self.list_to_string(t_prime_list)
        t_prime_prec = copy.deepcopy(t_prec)
        t_prime_prec.at[nt_ind, 'from'] = nt
        t_prime_prec.at[nt_ind, 'to'] = new_p
        t_prime_prec.at[nt_ind, 'toix'] = float(p_ind)
        t_prime_prec.at[nt_ind, 'li'] = new_prod_prob

        if nt == "S" or nt == "A":
            t_prime_rule = self.list_to_string(t_prime_list)
            t_prime_prec = t_prime_prec.loc[:nt_ind-cut:]
            return {"t_prime_rule": t_prime_rule, "t_prime_bv": t_prime_bv, "t_prime_prec": t_prime_prec, "nt_ind": nt_ind, "nt": nt}


        return {"t_prime_rule": t_prime_rule, "t_prime_bv": t_prime_bv, "t_prime_prec": t_prime_prec, "t_prime_list": t_prime_list}

