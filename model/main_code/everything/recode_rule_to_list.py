"""
Created on Sun Aug 10 16:19:00 2020

@author: jan-philippfranken
"""
###############################################################################
########################### Rule Translator ###################################
###############################################################################

####################### General imports #######################################
# -

####################### Custom imports ########################################
#-

class rule_translator():
    '''rule translator takes a string as input and converts it into a list of lists, or vice versa'''
    def __init__self(self):
        self.__init__self(self)

    def string_to_list(self, char_string):
        '''transforms string into list of lists'''
        c_list= list(char_string)            # first creating list of characters
        # adding necessary details (commas, changing brackets, etc) using multiple steps to improve readability
        c_string = [''.join("',[" if c == '(' else ']' if c == ')' else "'X'" if c == "X" else ",':'," if c == ":" else c for c in char) for char in c_list]
        c_string = [''.join("'Z" if c == "Z" else c for c in char) for char in c_string]
        c_string = ''.join(c_string)

        # final edits specifically focused on the present grammar (max 3 bound variables)
        for ch in ['x1', 'x2', 'x3']:
            if ch in c_string:
                    c_string=c_string.replace(ch,"'"+ch+"'")

        for ch in ["lambda"]:
            if ch in c_string:
                c_string=c_string.replace(ch,"'lambda',")

        # print(self.list_to_string(c_string))

            return list(eval(c_string))

    def list_to_string(self, char_list):
        '''transforms list of lists into string using similar steps as above just opposite goal'''
        c_string = str(char_list)[1:len(str(char_list))-2]
        c_string = [''.join("(" if c == "[" else '),' if c == ']' else ':' if c == ",':'," else c for c in char) for char in c_string]
        c_string = ''.join(c_string)


        # final edits specifically focused on the present grammar (max 3 bound variables)
        for ch in ["'Z.exactly'", "'Z.atmost'", "'Z.atleast'", "'Z.exists'", "'Z.forall'", # quantifiers
                   "'Z.or_operator'", "'Z.and_operator'", "'Z.not_operator'", "'Z.equal'", # booleans
                    "'Z.lequal'", "'Z.grequal'", "'Z.less'", "'Z.greater'", "'Z.hor_operator'", # comparison
                   "'lambda'", "':'", "'X'"]: # other stuff
            if ch in c_string:
                c_string=c_string.replace(ch, eval(ch))

        # bound variables
        for ch in ["'x1'", "'x2'", "'x3'"]: # bound variables]: # other stuff
            if ch in c_string:
                c_string=c_string.replace(ch, eval(ch)+",")

        # getting rid of commas and white space
        for ch in [", ("]:
            if ch in c_string:
                c_string=c_string.replace(ch, "(")

        for ch in [", "]:
            if ch in c_string:
                c_string=c_string.replace(ch, " ")

        for ch in [", :"]:
            if ch in c_string:
                c_string=c_string.replace(ch, ":")

        # appending closing bracket
        c_string = c_string + ")"

        for ch in [",)"]:
            if ch in c_string:
                c_string=c_string.replace(ch, ")")

        for ch in ["1 X", "2 X", "3 X"]:
            if ch in c_string:
                c_string=c_string.replace(ch, ch[0] + "," + " X")

         #finally all possible features
        for ch in ["'colour'", "'size'", "'orientation'", "'grounded'", "D", "E"]:
            if ch in c_string:
                c_string=c_string.replace(ch, "," + ch)

        for ch in [" ,", " ,"]:       # grounded
            if ch in c_string:
                c_string=c_string.replace(ch, ",")

        for ch in [" "]:       # grounded
            if ch in c_string:
                c_string=c_string.replace(ch, "")

        for ch in ["lambda", ":"]:       # grounded
            if ch in c_string:
                c_string=c_string.replace(ch, ch + " ")

        for ch in [",X))"]:       # groƒunded
            if ch in c_string:
                c_string=c_string.replace(ch, ",X)")

        for ch in ["xN", "xO"]:       # groƒunded
            if ch in c_string:
                c_string=c_string.replace(ch, ch + ",")

        for ch in ["x1x1","x1x2","x2x1","x1x3","x3x1","x2x3","x3x2","x2x2","x3x3"]:       # groƒunded
            if ch in c_string:
                c_string=c_string.replace(ch, ch[:2] + "," + ch[2:] + ",")

        for ch in ["'Z", "'L", "'X", "'K", "'C", "'J", "'1","'2","'3"]:       # groƒunded
            if ch in c_string:
                c_string=c_string.replace(ch, ch[1])

        for ch in ["'B"]:       # groƒunded
            if ch in c_string:
                c_string=c_string.replace(ch, ch[1] + ",")


        for ch in ["'S)", "'S"]:       # groƒunded
            if ch in c_string:
                c_string=c_string.replace(ch, ch[1])

        for ch in [")X","CX", "SX","S1","S2","S3","C1","C2","C3",")1",")2",")3","CZ"]:       # groƒunded
            if ch in c_string:
                c_string=c_string.replace(ch, ch[0] + "," + ch[1])

        for ch in ["C'"]:       # groƒunded
            if ch in c_string:
                c_string=c_string.replace(ch, ch[0])

        for ch in ["B)Z","D)Z","E)Z","G)Z","H)Z","I)Z"]:       # groƒunded
            if ch in c_string:
                c_string=c_string.replace(ch, ch[:2] + ',' + ch[2])

        for ch in ["B)'","E)'","H)'","D)'","I)'", "G)'"]:       # groƒunded
            if ch in c_string:
                c_string=c_string.replace(ch, ch[:2])


        for ch in [",,"]:       # groƒunded
            if ch in c_string:
                c_string=c_string.replace(ch, ",")




        return c_string


    def flatten(self, l):
        ''' this one flattens an arbitraryly deep nested list'''
        stack = [enumerate(l)]
        path = [None]
        while stack:
            for path[-1], x in stack[-1]:
                if isinstance(x, list):
                    stack.append(enumerate(x))
                    path.append(None)
                else:
                    yield x, tuple(path)
                break
            else:
                stack.pop()
                path.pop()

    def get_inds(self, l):
        '''this one gives the indices and values of an arbi'''
        all_ind = []
        for entry in self.flatten(l):
            all_ind.append(entry[0])
            all_ind.append(list(entry[1]))
        return  all_ind

    def get_list(self, l):
        '''this one gives the indices and values of an arbi'''
        flat_list = []
        for entry in self.flatten(l):
            flat_list.append(entry[0])
            # all_ind.append(list(entry[1]))
        return flat_list


# init_rule = "G.exists(lambda x1: G.exists(lambda x2: G.and_operator(G.and_operator(G.equal(x1,'blue','colour'),G.equal(x2,'red','colour')),G.hor_operator(x1,x2,'contact')),X),X)"

# cat = rule_translator()
# k = cat.string_to_list(init_rule)
# print(k)
# # print(k)
# # print(k)
# a = cat.list_to_string(k)
#
#
#
