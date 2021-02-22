"""
Created on Sun Mar 15 16:10:50 2020

@author: jan-philippfranken
"""
###############################################################################
####################### FUNCTIONS FOR PRODUCTIONS #############################
###############################################################################

class rules():
    def __init__self(self):
        self.__init__self(self)

    def exists(self, func, x):
        """existential operator (e.g., there is a small red cone, formally specified as
        E(lamda x1: and(=(x1,red,colour),=(x1,small,size),X))"""
        return any([func(i) for i in x])

    def forall(self, func, x):
        """ universal operator """
        return all([func(i) for i in x])

    def exactly(self, func, n, x):
        """ exact operator """
        if sum([func(i) for i in x]) == n:
            return True
        else:
            return False

    def atleast(self, func, n, x):
        """ at least operator """
        if sum([func(i) for i in x]) >= n:
            return True
        else:
            return False

    def atmost(self, func, n, x):
        """ at most operator """
        if sum([func(i) for i in x]) <= n:
            return True
        else:
            return False

    def and_operator(self, x, y):
        """ logical operator """
        return x and y

    def or_operator(self, x, y):
        """ logical operator """
        return x or y

    def hor_operator(self, x, y, dim):
        """ logical operator that checks if objects touch each other"""
        if x["id"] in y[dim]:
            return True
        else:
            return False

    def not_operator(self, x):
        """ logical operator """
        return not x

    def equal(self, x, y, dim):
        """ logical operator """
        if isinstance(y,list) ==  False:
            return x[dim] == y
        else:
            return x[dim] == y[dim]

    def grequal(self, x, y, dim):
        """ logical operator """
        if isinstance(y,list) ==  False:
            return x[dim] == y
        else:
            return x[dim] == y[dim]

    def lequal(self, x, y, dim):
        """ logical operator """
        if isinstance(y,list) ==  False:
            return x[dim] == y
        else:
            return x[dim] == y[dim]

    def greater(self, x, y, dim):
        """ logical operator """
        if isinstance(y,list) ==  False:
            return x[dim] == y
        else:
            return x[dim] == y[dim]

    def less(self, x, y, dim):
        """ logical operator """
        if isinstance(y,list) ==  False:
            return x[dim] == y
        else:
            return x[dim] == y[dim]

###############################################################################
###############################################################################
