"""
Created on Sun Mar 15 16:10:50 2020
import
@author: jan-philippfranken
"""

###############################################################################
####################### FUNCTIONS FOR PRODUCTIONS #############################
###############################################################################
from recode_rule_to_list import rule_translator
import numpy as np
# apply

# the class rules provides the logical operations that are used in the grammar
class rules(rule_translator):
    def __init__self(self):
        self.__init__self(self)

    def exists(self, func, x):
        """existential operator (e.g., there is a small red cone, formally specified as
        E(lamda x1: and(=(x1,red,colour),=(x1,small,size),X))"""
        a = any(list(map(func,x)))
        b = a
        c = list(map(func,x))
#         # print('hi')
#         print(a,b, c)
        return b

    def forall(self, func, x):
        """ universal operator. includes self-comparison (i.e. each object will be compared to all others
        including iteself) """
        a = all(list(map(func,x)))
        b = a
        c = list(map(func,x))
#         # print(a,b,c)
        return b

    def exactly(self, func, n, x):
        """ exact quantifier """
        a = sum(list(map(func,x)))
        b = a == n
        c = list(map(func,x))
        # print(a,n,b,c)
        return b

    def atleast(self, func, n, x):
        """ at least quantifier """
        a = sum(list(map(func,x)))
        b = a >= n
        c = list(map(func,x))
# #         # print(a,n,b,c)
        return b

    def atmost(self, func, n, x):
        a = sum(list(map(func,x)))
        b = a <= n
        c = list(map(func,x))

        if b == True and a >= 1:
            # print(a,n,b,c)
            return True
        else:
            b = False
            # print(a,n,b,c)
            return False

        # return b
        # return False

    def and_operator(self, x, y):
        """ and operator """
        return x and y

    def or_operator(self, x, y):
        """ or operator """
        if x and y:
            return x and y
        else:
            return x or y

    def hor_operator(self, x, y, dim):
        """ logical operator that checks if objects touch each other. note y[dim] needs to be longer than 1
        to prevent self comparison from counting as true (otherwise it will always be true)"""
        if x["id"] in y[dim] and len(y[dim]) > 1:
            return True
        else:
            return False

    def not_operator(self, x):
        """ not operator """
        return not x

    def equal(self, x, y, dim):
        """ equal operator """
        if x == y:
            return True
        if isinstance(y,dict) ==  False:

            return x[dim] == y
        else:
            return x[dim] == y[dim]

    def grequal(self, x, y, dim):
        """ greater or equal operator """
        if isinstance(y,dict) ==  False:
            return x[dim] >= y
        else:
            return x[dim] >= y[dim]

    def lequal(self, x, y, dim):
        """ less or equal operator """
        if isinstance(y,dict) ==  False:
            return x[dim] <= y
        else:
            return x[dim] <= y[dim]

    def greater(self, x, y, dim):
        """ greater operator, if x == y, it returns true since this is refers comparison"""
        if x == y:
            return True
        if isinstance(y,dict) ==  False:
            return x[dim] > y
        else:
            return x[dim] > y[dim]

    def less(self, x, y, dim):
        """ less operator, if x == y, it returns true since this refers to self comparison"""
        if x == y:
            return True
        if isinstance(y,dict) ==  False:
            return x[dim] < y
        else:
            return x[dim] < y[dim]

###############################################################################
###############################################################################
