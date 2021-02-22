import numpy as np
from recode_rule_to_list import rule_translator
import pandas as pd
import copy

####################### grammar ##############################################
S = ['Z.exists', 'Z.forall', 'Z.atleast', 'Z.atmost', 'Z.exactly']
A = ['B', 'S']
B = ['C', 'J(B,B)', 'Z.not_operator(B)']
C = ['Z.equal(xN, D)', 'K(xN, E)', 'Z.equal(xN,xO,G)', 'K(xN, xO, H)', 'Z.hor_operator(xN,xO,I)']
D = {"colour": ["'red'", "'blue'", "'green'"], "size": [1, 2, 3], "xpos": np.arange(9),"ypos": np.arange(2, 6), "rotation": np.arange(0, 6.5, 0.5),"orientation": ["'upright'", "'lhs'", "'rhs'", "'strange'"], "grounded": ["'no'", "'yes'"]}
E = {"size": [1, 2, 3], "xpos": np.arange(9), "ypos": np.arange(2, 6), "rotation": np.arange(0, 6.3, 0.1)}
G = ["'colour'", "'size'", "'xpos'", "'ypos'", "'rotation'", "'orientation'", "'grounded'"]
H = ["'size'", "'xpos'", "'ypos'", "'rotation'"]
I = ["'contact'"]
J = ['Z.and_operator', 'Z.or_operator']
K = ['Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']
L = ['Z.atleast', 'Z.atmost', 'Z.exactly']
M = [1, 2, 3]

# summarizing grammar in dictionary
productions = {"S": S, "A": A, "B": B, "C": C, "D": D, "E": E, "G": G, "H": H, "I": I, "J": J, "K": K, "L": L, "M": M}

# probabilities:
Swin =  {'Z.exists': 1/3, 'Z.forall': 1/3, 'L': 1/3, "A": 1}
Awin = {"B": 1/2, "S": 1/2}
Bwin = {'C': 1/3, 'J(B,B)': 1/3, 'Z.not_operator': 1/3}
Cwin = {'D': .2, 'E': .2, 'G': .2, 'H': .2, 'I': .2}
Dwin =  {"colour": {"feat": .25, "values": {"red": 1/3, "blue": 1/3, "green": 1/3}},
         "size": {"feat": .25, "values": {"1": 1/3, "2": 1/3, "3": 1/3}},
         "orientation": {"feat": .25, "values": {"upright": 1/4, "lhs": 1/4, "rhs": 1/4, "strange": 1/4}},
         "grounded": {"feat": .25, "values": {"no": .5, "yes": .5}}}
Ewin = {"size": {"feat": 1, "values": {"1": 1/3, "2": 1/3, "3": 1/3}}}
Gwin = {'colour': .25, 'size': .25, 'xpos': 0, 'ypos': 0, 'rotation': 0, 'orientation': .25, 'grounded': .25}
Hwin = {"size": 1}
Iwin = {"contact": 1}
Jwin =  {"Z.and_operator":.5,"Z.or_operator":.5}
Kwin =  {'Z.lequal': .25, 'Z.grequal': .25, 'Z.less': .25, 'Z.greater': .25}
Lwin =  {'Z.atleast': 1/3, 'Z.atmost': 1/3, 'Z.exactly': 1/3}
Mwin = {"1": 0.333333333,"2": 0.333333333, "3": 0.333333333}

# summarizing probs in dictionary
probabilities = {"S": Swin, "A": Awin, "B": Bwin, "C": Cwin,
                 "D": Dwin, "E": Ewin, "G": Gwin, "H": Hwin, "I": Iwin,
                 "J": Jwin, "K": Kwin, "L": Lwin, "M": Mwin}

class reverse_rule(rule_translator):
    def __init__self(self):
        self.__init__self(self)

# exmpl = ['Z.forall', ['lambda', 'x1', ':', 'Z.not_operator', ['Z.not_operator', ['Z.or_operator', ['Z.and_operator', ['Z.or_operator', ['Z.not_operator', ['Z.not_operator', ['Z.equal', ['x1', 'green', 'colour']]], 'Z.not_operator', ['Z.equal', ['x1', 'green', 'colour']]], 'Z.equal', ['x1', 'blue', 'colour']], 'Z.not_operator', ['Z.equal', ['x1', 'blue', 'colour']]]]], 'X']]
# print(exmpl)
# t_prec = pd.read_csv('tprec.csv')
# print(t_prec)
# rr = reverse_rule()
# flist = rr.get_inds(exmpl)
# # print(flist)

    def get_prec_recursively(self, l):
        flist = self.get_inds(l)
        to = {}
        for k in productions.keys():
            to[k] = []

        to["Ef"] = []
        to["Ev"] = []
        from_nt = []
        to_nt =[]
        li = []
        flist_copy = copy.deepcopy(flist)
        # first checking for "S" and "A"

        while any([i for i in flist if(i in ['Z.exists', 'Z.forall', 'Z.atleast', 'Z.atmost', 'Z.exactly'])]):
            for i in flist:
                if i in ['Z.exists', 'Z.forall', 'Z.atleast', 'Z.atmost', 'Z.exactly']:

                    if flist[flist.index(i) + 8 * max(1, len(to["S"]))] in ['Z.exists', 'Z.forall', 'Z.atleast', 'Z.atmost', 'Z.exactly']:
                        to["A"].append(["S", probabilities["A"]["S"]])
                        li.append(probabilities["A"]["S"])
                        # print('hi')
                    else:
                        to["A"].append(["B", probabilities["A"]["B"]])
                        li.append(probabilities["A"]["B"])

                if i in ['Z.exists', 'Z.forall']:
                    to["S"].append([i, probabilities["S"][i]])

                    # cut = 0
                    # if len(to["A"]) >= 1:
                    #     cut = 1
                    to_nt.append(to["A"][len(to["A"])-1][0])
                    from_nt.append("S")
                    from_nt.append("A")
                    to_nt.append(i)
                    li.append(probabilities["S"][i])



                if i in ['Z.atleast', 'Z.atmost', 'Z.exactly']:
                    # if len(to["S"]) <= 2:
                    #     to["A"].append(["S", probabilities["A"]["S"]])
                        # to_nt.append(to["A"][len(to["A"])-1][0])
                    to["S"].append(["L", probabilities["S"]["L"]])
                    to["L"].append([i, probabilities["L"][i]])

                    from_nt.append("S")
                    from_nt.append("A")
                    from_nt.append("L")
                    to_nt.append('L')
                    to_nt.append(i)
                    li.append(probabilities["S"]["L"])
                    li.append(probabilities["L"][i])


                    # cut = 0
                    # if len(to["A"]) >= 1:
                    #     cut = 1
                    # print(cut)
                    # print(to["A"])
                    to_nt.append(to["A"][len(to["A"])-1][0])

                if i in [1,2,3] and flist[flist.index(i)+2] in ['X'] and flist.index(i) >= len(flist) - flist.index('X'):
                    to["M"].append([i, probabilities["M"][str(i)]])
                    from_nt.append("M")
                    to_nt.append(i)
                    li.append(probabilities["M"][str(i)])

            break

        # to_nt.append("B")

        # now B
        while any([i for i in flist if(i in ['Z.and_operator', 'Z.or_operator', 'Z.not_operator'])]):
            for i in flist:
                if i in ['Z.and_operator', 'Z.or_operator']:
                    if len(to["B"]) == 0:
                        to["A"].append(["B", probabilities["A"]["B"]])
                    to['B'].append(["J(B,B)", probabilities['B']["J(B,B)"]])
                    li.append(probabilities["B"]["J(B,B)"])

                    from_nt.append("B")
                    from_nt.append("J")
                    to_nt.append('J(B,B)')
                    if i == 'Z.and_operator':
                        to['J'].append(['Z.and_operator', probabilities['J']["Z.and_operator"]])
                        to_nt.append('Z.and_operator')
                        li.append(probabilities["J"][i])
                    if i == 'Z.or_operator':
                        to['J'].append(['Z.or_operator', probabilities['J']["Z.or_operator"]])
                        to_nt.append('Z.or_operator')
                        li.append(probabilities["J"][i])
                if i in ['Z.not_operator']:
                    from_nt.append("B")
                    to_nt.append('not_operator(B)')
                    li.append(probabilities["B"][i])
                    if len(to["B"]) == 0:
                        to["A"].append(["B", probabilities["A"]["B"]])
                    to['B'].append(['Z.not_operator', probabilities['B']['Z.not_operator']])
            break

        # now C
        while any([i for i in flist if(i in ['Z.equal','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater','Z.hor_operator'])]):
            for i in flist:
                if i in ['Z.equal']:
                    from_nt.append("B")
                    from_nt.append("C")
                    to_nt.append("C")
                    li.append(probabilities["B"]["C"])

                    to["B"].append(["C", probabilities["B"]["C"]])
                    if flist[flist.index(i)+4] not in ['x1', 'x2', 'x3']: # make sure this does not involve two bound variables
                        to["C"].append(["D", probabilities["C"]["D"]])
                        to_nt.append('Z.equal(xN,D)')
                        li.append(probabilities["C"]["D"])
                    else:
                        to["C"].append(["G", probabilities["C"]["G"]])
                        to_nt.append('Z.equal(xN,xO,G)')
                        li.append(probabilities["C"]["G"])

                if i in ['Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']:
                    li.append(probabilities["B"]["C"])
                    to_nt.append("C")
                    from_nt.append("K")
                    to_nt.append(i)
                    from_nt.append("B")
                    from_nt.append("C")
                    to["B"].append(["C", probabilities["B"]["C"]])
                    to["K"].append([i, probabilities["K"][i]])
                    if flist[flist.index(i)+4] not in ['x1', 'x2', 'x3']: # make sure this does not involve two bound variables
                        to["C"].append(["E", probabilities["C"]["E"]])
                        to_nt.append('K(xN, E)')
                        li.append(probabilities["C"]["E"])
                        li.append(probabilities["K"][i])

                    else:
                        to["C"].append(["H", probabilities["C"]["H"]])
                        to_nt.append('K(xN, xO, H)')
                        li.append(probabilities["C"]["H"])
                        li.append(probabilities["K"][i])

                if i in ['Z.hor_operator']:
                    li.append(probabilities["B"]["C"])
                    li.append(probabilities["C"]["I"])
                    to_nt.append("C")
                    li.append(probabilities["I"]["contact"])
                    to_nt.append("Z.hor_operator(xN,xO,I)")
                    from_nt.append("B")
                    from_nt.append("C")
                    to["B"].append(["C", probabilities["B"]["C"]])
                    to["C"].append(["I", probabilities["C"]["I"]])
                    to["I"].append([productions["I"], probabilities["I"]["contact"]])
                    to_nt.append('contact')
                    from_nt.append('Ef')
            break

        # now features and values
        while any([i for i in flist if(i in ['colour', 'size','xpos', 'ypos', 'rotation', 'orientation', 'grounded'])]):
            for i in flist:
                if i in ['colour','size','xpos', 'ypos', 'rotation', 'orientation','grounded']:
                    if flist[flist.index(i)-2] not in ['x1', 'x2', 'x3']:
                        if flist[flist.index(i)-6] in ['Z.equal']:
                            to["Ef"].append([i, probabilities["D"][i]['feat']])
                            to["Ev"].append([flist[flist.index(i)-2], probabilities["D"][i]['values'][str(flist[flist.index(i)-2])]])
                            li.append(probabilities["D"][i]["feat"])
                            from_nt.append("Ef")
                            from_nt.append("Ev")
                            to_nt.append(i)
                            to_nt.append(flist[flist.index(i)-2])
                            li.append(probabilities["D"][i]["values"][str(flist[flist.index(i)-2])])
                        if flist[flist.index(i)-6] in ['Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']:
                            li.append(probabilities["E"][i]["feat"])
                            to["Ef"].append([i, probabilities["E"][i]['feat']])
                            to["Ev"].append([flist[flist.index(i)-2], probabilities["E"][i]['values'][str(flist[flist.index(i)-2])]])
                            li.append(probabilities["E"][i]["values"][str(flist[flist.index(i)-2])])
                            from_nt.append("Ef")
                            from_nt.append("Ev")
                            to_nt.append(i)
                            to_nt.append(flist[flist.index(i)-2])
                    if flist[flist.index(i)-2] in ['x1', 'x2', 'x3']:
                        if flist[flist.index(i)-6] in ['Z.equal']:
                            to["Ef"].append([i, probabilities["G"][i]])
                            li.append(probabilities["G"][i])
                            from_nt.append("Ef")
                            to_nt.append(i)
                        if flist[flist.index(i)-6] in ['Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']:
                            to["Ef"].append([i, probabilities["H"][i]])
                            from_nt.append("Ef")
                            li.append(probabilities["H"][i])
                            to_nt.append(i)





                    flist.remove(i)
                    # if flist[flist.index(i)-2] in ['x1', 'x2', 'x3']:
                    #     print(i)
            break

        to = {k:v for (k,v) in to.items() if v != []}
        # print(to)z
        # # from_nt = list(to.keys())
        # # from_nt.remove('B')
#         # print(len(from_nt))
#         # print(to_nt)
#         # print(len(from_nt))
#         # print(len(to_nt))

        t_prime_prec = pd.DataFrame({"from": from_nt, "to": to_nt, "li": li})

        return t_prime_prec
