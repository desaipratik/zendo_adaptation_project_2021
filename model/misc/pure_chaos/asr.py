t_prime_list = ['Z.exists', ['lambda', 'x1', ':', 'Z.exists', ['lambda', 'x2', ':', 'Z.exactly', ['lambda', 'x3', ':', 'Z.or_operator', ['Z.not_operator', ['Z.grequal', ['x2', 3, 'size']], 'Z.or_operator', ['Z.not_operator', ['Z.not_operator', ['Z.lequal', ['x2', 2, 'size']]], ['Z.not_operator', ['Z.greater', ['x2', 'x1', 'size']]]]], 1, 'X'], 'X'], 'X']]

spec_ind_s = '[1][4][4][4][3][2]'
print(eval('t_prime_list' + spec_ind_s))
# ['Z.hor_operator', ['x1', 'x1', 'contact']]
spec_ind_check = spec_ind_s[:-3]
print(spec_ind_check)
# [1][4][1]
print(eval('t_prime_list' + spec_ind_check))
# # ['Z.not_operator', ['Z.hor_operator', ['x1', 'x1', 'contact']], ['Z.hor_operator', ['x1', 'x1', 'contact']]]
def insertList(listToFlatten, outerList, LIST, insind):
    insind2=0
    for item in listToFlatten:
        outerList.insert(insind+insind2,item)
        insind2+=1
    outerList.remove(listToFlatten)
mid_obj_ind = spec_ind_s[:-3] + str([int(spec_ind_s[-2])-1])
print(eval('t_prime_list'+mid_obj_ind))
print('asd')
print(mid_obj_ind)
print(t_prime_list)
insertList(eval('t_prime_list' + spec_ind_s),eval('t_prime_list' + spec_ind_check),t_prime_list,int(spec_ind_s[-2]))
print(t_prime_list)
# CRAVEr
# 0
# 2
# meowing
# vat in the houst
# 0
# 2
# 0
# hacerino
# ['Z.exactly', ['lambda', 'x1', ':', 'Z.and_operator', ['Z.and_operator', ['Z.not_operator', 'Z.hor_operator', ['x1', 'x1', 'contact'], ['Z.hor_operator', ['x1', 'x1', 'contact']]], 'Z.equal', ['x1', 'x1', 'orientation']], 2, 'X']]
# results
# Z.exactly(lambda x1: Z.and_operator(Z.and_operator(Z.not_operator(Z.hor_operator(x1,x1,'contact')),Z.not_operator(Z.hor_operator(x1,x1,'contact'))),Z.equal(x1,x1,'orientation')),2,X)
# Z.exactly(lambda x1: Z.and_operator(Z.and_operator(Z.not_operatorZ.hor_operator(x1,x1,'contact'),(Z.hor_operator(x1,x1,'contact'))),Z.equal(x1,x1,'orientation')),2,X)
