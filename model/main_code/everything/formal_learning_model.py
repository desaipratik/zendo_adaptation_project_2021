"""
Created on Fri Mar 13 11:41:11 2020

@author: jan-philippfranken
"""

# libraries
import numpy as np
import math

# P(F|G) given by a dirichlet multinomial PCFG
# input is a list of lists (i.e., all non-terminals), where each element entails
# information about the number of each possible extensions of a non-terminal symbol
class prior():
    def __init__self(self):
        self.__init__self(self)

    def compute_norm(self, c_n):
        """ computing normalizing constant """
        c_gamma = []
        for i in c_n:
            c_gamma.append(math.gamma(i))
        return np.prod(c_gamma) / math.gamma(np.sum([c_n]))

    def compute_prior(self, c_a_h):
        """ computing prior for single production step """
        d_a_h = []
        for i in c_a_h:
            d_a_h.append(1)
        c_a_h = np.array(c_a_h)
        d_a_h = np.array(d_a_h)
        return self.compute_norm(c_a_h + d_a_h) / self.compute_norm(d_a_h)

    def prod_product(self, non_terminals):
        """ returning product of productions for all non terminal symbols """
        return np.prod([self.compute_prior(i) for i in non_terminals])

# likelihood e^-bQ_l(F)
class likelihood():
    def __int__self(self):
        self.__init__self(self)
    def compute_ll(self, out_prob, out_count):
        return math.e**(-(out_prob * out_count))

class posterior():
    def __init__self(self):
        self.__init__self(self)
        self.prior = prior()
        self.ll = likelihood()
    def process_post(self, prior, ll):
        return prior * ll
