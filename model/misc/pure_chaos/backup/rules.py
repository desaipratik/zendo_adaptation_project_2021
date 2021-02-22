"""
Created on Sun Mar 15 16:10:50 2020

@author: jan-philippfranken
"""
###############################################################################
####################### FUNCTIONS FOR PRODUCTIONS #############################
###############################################################################

# the class rules provides the logical operations that are used in the grammar

class rules():
    def __init__self(self):
        self.__init__self(self)

    def exists(self, func, x):
        """existential operator (e.g., there is a small red cone, formally specified as
        E(lamda x1: and(=(x1,red,colour),=(x1,small,size),X))"""
        return any([func(i) for i in x])

    def forall(self, func, x):
        """ universal operator. includes self-comparison (i.e. each object will be compared to all others
        including iteself) """
        return all([func(i) for i in x])

    def exactly(self, func, n, x):
        """ exact quantifier """
        if sum([func(i) for i in x]) == n:
            return True
        else:
            return False

    def atleast(self, func, n, x):
        """ at least quantifier """
        if sum([func(i) for i in x]) >= n:
            return True
        else:
            return False

    def atmost(self, func, n, x):
        """ at most quantifier """
        if sum([func(i) for i in x]) <= n:
            return True
        else:
            return False

    def and_operator(self, x, y):
        """ and operator """
        return x and y

    def or_operator(self, x, y):
        """ or operator """
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


