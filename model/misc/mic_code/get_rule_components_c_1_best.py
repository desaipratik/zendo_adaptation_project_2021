"""
Created on Sun Mar 15 15:01:21 2020

@author: jan-philippfranken
"""
###############################################################################
########################### Main File #########################################
###############################################################################


####################### General imports #######################################
import numpy as np
import pandas as pd
import random
import copy


####################### Custom imports ########################################
from pcfg_generator import pcfg_generator
from tree_regrower import tree_regrower
from reverse_rule import reverse_rule




###################### Preliminaries ##########################################
Z = pcfg_generator()                                         # instantiating pcfg generator
tg = tree_regrower()
rr = reverse_rule()
random.seed(1)                                                # setting random.seed to allow replication of mcmc chain
main_data_formatted = pd.read_csv('model_results/normative_res_one_0_process_models_only_1kbest.csv')  # getting the preprocessed data file



main_data_formatted = main_data_formatted.query("post_resp != 's'")
main_data_formatted = main_data_formatted.reset_index(drop=True)
# removing all complex rules from the data frame, allowing only the five simple booleans to remain (Zeta, Upsilon, Iota, Kappa, Omega)
main_data_formatted = main_data_formatted.query("rule_name == 'Zeta' or rule_name == 'Upsilon' or rule_name == 'Iota' or rule_name == 'Kappa' or rule_name == 'Omega'")
main_data_formatted = main_data_formatted.reset_index(drop=True)
# creating dictionary with subject's tokens as keys and the number of trials each subject completed after removing complex rules as values
trial_counts = main_data_formatted.groupby('token_id').size()
trial_counts = dict(trial_counts)
# print(trial_counts)
# print(sum(list(trial_counts.values())))


# getting rule components in frequenceis
quantifiers_r = []
booleans_r = []
equalities_r = []
features_r = []
relations_r = []


# getting exact rule components
q_r_specific = []
b_r_specific = []
e_r_specific = []
f_r_specific = []
r_r_specific = []

init_probs = []
rev_probs = []
for rrrs in main_data_formatted['post_resp'][:248]:
  rev_probs.append(np.log(np.prod(rr.get_prec_recursively(rr.string_to_list(rrrs))['li'])))
  # frequencies
  from_list = list(rr.get_prec_recursively(rr.string_to_list(rrrs))['from'])
  quantifiers_r.append(from_list.count('S'))
  bools = from_list.count('B') - from_list.count('C')
  booleans_r.append(bools)
  equalities_r.append(from_list.count('C'))
  to_list = list(rr.get_prec_recursively(rr.string_to_list(rrrs))['to'])
  relations_r.append(to_list.count('Z.hor_operator(xN,xO,I)'))
  features_r.append(from_list.count('Ef')-to_list.count('Z.hor_operator(xN,xO,I)'))

  # specifics
  quants = [i for i in to_list if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
  bools_s = [i for i in to_list if i in ['J(B,B)', 'not_operator(B)']]
  equals = [i for i in to_list if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
  feats = [i for i in to_list if i in ['colour','size','orientation','grounded']]
  reals = [i for i in to_list if i in ['contact']]
  q_r_specific.append(quants)
  b_r_specific.append(bools_s)
  e_r_specific.append(equals)
  f_r_specific.append(feats)
  r_r_specific.append(reals)




#

# getting rule components
quantifiers_i = []
booleans_i = []
equalities_i = []
features_i = []
relations_i = []

# getting exact rule components
q_i_specific = []
b_i_specific = []
e_i_specific = []
f_i_specific = []
r_i_specific = []


# # print(rev_probs)
# # print(np.mean(rev_probs))
for rrr in main_data_formatted['prior_resp'][:248]:
  init_probs.append(np.log(np.prod(rr.get_prec_recursively(rr.string_to_list(rrr))['li'])))
  from_list = list(rr.get_prec_recursively(rr.string_to_list(rrr))['from'])
  quantifiers_i.append(from_list.count('S'))
  bools = from_list.count('B') - from_list.count('C')
  booleans_i.append(bools)
  equalities_i.append(from_list.count('C'))
  to_list = list(rr.get_prec_recursively(rr.string_to_list(rrr))['to'])
  features_i.append(from_list.count('Ef')-to_list.count('Z.hor_operator(xN,xO,I)'))
  relations_i.append(to_list.count('Z.hor_operator(xN,xO,I)'))

  # specifics
  quants = [i for i in to_list if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
  bools_s = [i for i in to_list if i in ['J(B,B)', 'not_operator(B)']]
  equals = [i for i in to_list if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
  feats = [i for i in to_list if i in ['colour','size','orientation','grounded']]
  reals = [i for i in to_list if i in ['contact']]
  q_i_specific.append(quants)
  b_i_specific.append(bools_s)
  e_i_specific.append(equals)
  f_i_specific.append(feats)
  r_i_specific.append(reals)

# # print(equalities_i[:5])
# # print(list(main_data_formatted['prior_resp'][:5]))


############### wsls learner
# now for process modles
# getting exact rule components
q_r_specific_wsls = []
b_r_specific_wsls = []
e_r_specific_wsls = []
f_r_specific_wsls = []
r_r_specific_wsls = []

q_added_all = []
q_shared_all = []
q_removed_all = []
q_difference_all = []

b_added_all = []
b_shared_all = []
b_removed_all = []
b_difference_all = []

e_added_all = []
e_shared_all = []
e_removed_all = []
e_difference_all = []

f_added_all = []
f_shared_all = []
f_removed_all = []
f_difference_all = []

r_added_all = []
r_shared_all = []
r_removed_all = []
r_difference_all = []

init_probs_wsls = init_probs
rev_probs_wsls = []
wsls_rules = []

with open('model_results/test4/rules_post_all_chain.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()
    for line in filecontents:
        # remove linebreak which is the last character of the string
        current_place = line[:-1]
        # add item to the list
        wsls_rules.append(current_place)


for wsls_rule_list,init_resp in zip(wsls_rules[:248],main_data_formatted['prior_resp'][:248]):
  rev_probs_wsls_all = []
  q_r_specific_wsls_all = []
  b_r_specific_wsls_all = []
  e_r_specific_wsls_all = []
  f_r_specific_wsls_all = []
  r_r_specific_wsls_all = []

  to_list_i = list(rr.get_prec_recursively(rr.string_to_list(init_resp))['to'])
  quants_i = [i for i in to_list_i if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
  bools_i = [i for i in to_list_i if i in ['J(B,B)', 'not_operator(B)']]
  equals_i = [i for i in to_list_i if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
  feats_i = [i for i in to_list_i if i in ['colour','size','orientation','grounded']]
  reals_i = [i for i in to_list_i if i in ['contact']]

  q_added_s = []
  q_shared_s = []
  q_removed_s = []
  q_difference_s = []

  b_added_s = []
  b_shared_s = []
  b_removed_s = []
  b_difference_s = []

  e_added_s = []
  e_shared_s = []
  e_removed_s = []
  e_difference_s = []

  f_added_s = []
  f_shared_s = []
  f_removed_s = []
  f_difference_s = []

  r_added_s = []
  r_shared_s = []
  r_removed_s = []
  r_difference_s = []

  for wsls_rule in eval(wsls_rule_list):
    rev_probs_wsls_all.append(np.log(np.prod(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['li'])))
    from_list = list(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['from'])
    to_list = list(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['to'])



    quants_r = [i for i in to_list if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
    bools_r = [i for i in to_list if i in ['J(B,B)', 'not_operator(B)']]
    equals_r = [i for i in to_list if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
    feats_r = [i for i in to_list if i in ['colour','size','orientation','grounded']]
    reals_r = [i for i in to_list if i in ['contact']]

    q_shared = sum([a==b for a,b in zip(quants_i,quants_r) if len(quants_i)==len(quants_r)])
    q_added=0
    q_removed=0
    if len(quants_i) > len(quants_r):
      q_removed = len(quants_i) - len(quants_r)
    elif len(quants_i) < len(quants_r):
      q_added = len(quants_r) - len(quants_i)
    q_difference = sum([a!=b for a,b in zip(quants_i,quants_r) if len(quants_i)==len(quants_r)])

    q_shared_s.append(q_shared)
    q_added_s.append(q_added)
    q_removed_s.append(q_removed)
    q_difference_s.append(q_difference)

    b_shared = sum([a==b for a,b in zip(bools_i,bools_r) if len(bools_i)==len(bools_r)])
    b_added=0
    b_removed=0
    if len(bools_i) > len(bools_r):
      b_removed = len(bools_i) - len(bools_r)
    elif len(bools_i) < len(bools_r):
      b_added = len(bools_r) - len(bools_i)
    b_difference = sum([a!=b for a,b in zip(bools_i,bools_r) if len(bools_i)==len(bools_r)])

    b_shared_s.append(b_shared)
    b_added_s.append(b_added)
    b_removed_s.append(b_removed)
    b_difference_s.append(b_difference)



    e_shared = sum([a==b for a,b in zip(equals_i,equals_r) if len(equals_i)==len(equals_r)])
    e_added=0
    e_removed=0
    if len(equals_i) > len(equals_r):
      e_removed = len(equals_i) - len(equals_r)
    elif len(equals_i) < len(equals_r):
      e_added = len(equals_r) - len(equals_i)
    e_difference = sum([a!=b for a,b in zip(equals_i,equals_r) if len(equals_i)==len(equals_r)])

    e_shared_s.append(e_shared)
    e_added_s.append(e_added)
    e_removed_s.append(e_removed)
    e_difference_s.append(e_difference)


    f_shared = sum([a==b for a,b in zip(feats_i,feats_r) if len(feats_i)==len(feats_r)])
    f_added=0
    f_removed=0
    if len(feats_i) > len(feats_r):
      f_removed = len(feats_i) - len(feats_r)
    elif len(feats_i) < len(feats_r):
      f_added = len(feats_r) - len(feats_i)
    f_difference = sum([a!=b for a,b in zip(feats_i,feats_r) if len(feats_i)==len(feats_r)])

    f_shared_s.append(f_shared)
    f_added_s.append(f_added)
    f_removed_s.append(f_removed)
    f_difference_s.append(f_difference)


    r_shared = sum([a==b for a,b in zip(reals_i,reals_r) if len(reals_i)==len(reals_r)])
    r_added=0
    r_removed=0
    if len(reals_i) > len(reals_r):
      r_removed = len(reals_i) - len(reals_r)
    elif len(reals_i) < len(reals_r):
      r_added = len(reals_r) - len(reals_i)
    r_difference = sum([a!=b for a,b in zip(reals_i,reals_r) if len(reals_i)==len(reals_r)])

    r_shared_s.append(r_shared)
    r_added_s.append(r_added)
    r_removed_s.append(r_removed)
    r_difference_s.append(r_difference)








    # q_r_specific_wsls_all.append(quants)
    # b_r_specific_wsls_all.append(bools_s)
    # e_r_specific_wsls_all.append(equals)
    # f_r_specific_wsls_all.append(feats)
    # r_r_specific_wsls_all.append(reals)



  rev_probs_wsls.append(np.mean(rev_probs_wsls_all))
  # q_r_specific_wsls.append(q_r_specific_wsls_all)
  # b_r_specific_wsls.append(b_r_specific_wsls_all)
  # e_r_specific_wsls.append(e_r_specific_wsls)
  # f_r_specific_wsls.append(f_r_specific_wsls)
  # r_r_specific_wsls.append(r_r_specific_wsls)

  q_shared_all.append(np.mean(q_shared_s))
  q_difference_all.append(np.mean(q_difference_s))
  q_added_all.append(np.mean(q_added_s))
  q_removed_all.append(np.mean(q_removed_s))


  b_shared_all.append(np.mean(b_shared_s))
  b_difference_all.append(np.mean(b_difference_s))
  b_added_all.append(np.mean(b_added_s))
  b_removed_all.append(np.mean(b_removed_s))


  e_shared_all.append(np.mean(e_shared_s))
  e_difference_all.append(np.mean(e_difference_s))
  e_added_all.append(np.mean(e_added_s))
  e_removed_all.append(np.mean(e_removed_s))


  f_shared_all.append(np.mean(f_shared_s))
  f_difference_all.append(np.mean(f_difference_s))
  f_added_all.append(np.mean(f_added_s))
  f_removed_all.append(np.mean(f_removed_s))

  r_shared_all.append(np.mean(r_shared_s))
  r_difference_all.append(np.mean(r_difference_s))
  r_added_all.append(np.mean(r_added_s))
  r_removed_all.append(np.mean(r_removed_s))

  # print('step')


############### tr learner
# now for process modles
# getting exact rule components
q_r_specific_wsls = []
b_r_specific_wsls = []
e_r_specific_wsls = []
f_r_specific_wsls = []
r_r_specific_wsls = []

q_added_all_tr = []
q_shared_all_tr = []
q_removed_all_tr = []
q_difference_all_tr = []

b_added_all_tr = []
b_shared_all_tr = []
b_removed_all_tr = []
b_difference_all_tr = []

e_added_all_tr = []
e_shared_all_tr = []
e_removed_all_tr = []
e_difference_all_tr = []

f_added_all_tr = []
f_shared_all_tr = []
f_removed_all_tr = []
f_difference_all_tr = []

r_added_all_tr = []
r_shared_all_tr = []
r_removed_all_tr = []
r_difference_all_tr = []

init_probs_tr= init_probs
rev_probs_tr = []
wsls_rules = []

with open('model_results/test4/rules_post_all_seed_chain_clean_1kbest.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()
    for line in filecontents:
        # remove linebreak which is the last character of the string
        current_place = line[:-1]
        # add item to the list
        wsls_rules.append(current_place)


for wsls_rule_list,init_resp in zip(wsls_rules[:248],main_data_formatted['prior_resp'][:248]):
  rev_probs_wsls_all = []
  q_r_specific_wsls_all = []
  b_r_specific_wsls_all = []
  e_r_specific_wsls_all = []
  f_r_specific_wsls_all = []
  r_r_specific_wsls_all = []

  to_list_i = list(rr.get_prec_recursively(rr.string_to_list(init_resp))['to'])
  quants_i = [i for i in to_list_i if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
  bools_i = [i for i in to_list_i if i in ['J(B,B)', 'not_operator(B)']]
  equals_i = [i for i in to_list_i if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
  feats_i = [i for i in to_list_i if i in ['colour','size','orientation','grounded']]
  reals_i = [i for i in to_list_i if i in ['contact']]

  q_added_s = []
  q_shared_s = []
  q_removed_s = []
  q_difference_s = []

  b_added_s = []
  b_shared_s = []
  b_removed_s = []
  b_difference_s = []

  e_added_s = []
  e_shared_s = []
  e_removed_s = []
  e_difference_s = []

  f_added_s = []
  f_shared_s = []
  f_removed_s = []
  f_difference_s = []

  r_added_s = []
  r_shared_s = []
  r_removed_s = []
  r_difference_s = []

  for wsls_rule in eval(wsls_rule_list):
    # print('hi')


    rev_probs_wsls_all.append(np.log(np.prod(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['li'])))
    from_list = list(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['from'])
    to_list = list(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['to'])

    # if wsls_rule != init_resp:



    quants_r = [i for i in to_list if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
    bools_r = [i for i in to_list if i in ['J(B,B)', 'not_operator(B)']]
    equals_r = [i for i in to_list if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
    feats_r = [i for i in to_list if i in ['colour','size','orientation','grounded']]
    reals_r = [i for i in to_list if i in ['contact']]

    q_shared = sum([a==b for a,b in zip(quants_i,quants_r) if len(quants_i)==len(quants_r)])
    q_added=0
    q_removed=0
    if len(quants_i) > len(quants_r):
      q_removed = len(quants_i) - len(quants_r)
    elif len(quants_i) < len(quants_r):
      q_added = len(quants_r) - len(quants_i)
    q_difference = sum([a!=b for a,b in zip(quants_i,quants_r) if len(quants_i)==len(quants_r)])

    q_shared_s.append(q_shared)
    # print(q_shared)
    q_added_s.append(q_added)
    q_removed_s.append(q_removed)
    q_difference_s.append(q_difference)

    b_shared = sum([a==b for a,b in zip(bools_i,bools_r) if len(bools_i)==len(bools_r)])
    b_added=0
    b_removed=0
    if len(bools_i) > len(bools_r):
      b_removed = len(bools_i) - len(bools_r)
    elif len(bools_i) < len(bools_r):
      b_added = len(bools_r) - len(bools_i)
    b_difference = sum([a!=b for a,b in zip(bools_i,bools_r) if len(bools_i)==len(bools_r)])

    b_shared_s.append(b_shared)
    b_added_s.append(b_added)
    b_removed_s.append(b_removed)
    b_difference_s.append(b_difference)



    e_shared = sum([a==b for a,b in zip(equals_i,equals_r) if len(equals_i)==len(equals_r)])
    e_added=0
    e_removed=0
    if len(equals_i) > len(equals_r):
      e_removed = len(equals_i) - len(equals_r)
    elif len(equals_i) < len(equals_r):
      e_added = len(equals_r) - len(equals_i)
    e_difference = sum([a!=b for a,b in zip(equals_i,equals_r) if len(equals_i)==len(equals_r)])

    e_shared_s.append(e_shared)
    e_added_s.append(e_added)
    e_removed_s.append(e_removed)
    e_difference_s.append(e_difference)


    f_shared = sum([a==b for a,b in zip(feats_i,feats_r) if len(feats_i)==len(feats_r)])
    f_added=0
    f_removed=0
    if len(feats_i) > len(feats_r):
      f_removed = len(feats_i) - len(feats_r)
    elif len(feats_i) < len(feats_r):
      f_added = len(feats_r) - len(feats_i)
    f_difference = sum([a!=b for a,b in zip(feats_i,feats_r) if len(feats_i)==len(feats_r)])

    f_shared_s.append(f_shared)
    f_added_s.append(f_added)
    f_removed_s.append(f_removed)
    f_difference_s.append(f_difference)


    r_shared = sum([a==b for a,b in zip(reals_i,reals_r) if len(reals_i)==len(reals_r)])
    r_added=0
    r_removed=0
    if len(reals_i) > len(reals_r):
      r_removed = len(reals_i) - len(reals_r)
    elif len(reals_i) < len(reals_r):
      r_added = len(reals_r) - len(reals_i)
    r_difference = sum([a!=b for a,b in zip(reals_i,reals_r) if len(reals_i)==len(reals_r)])

    r_shared_s.append(r_shared)
    r_added_s.append(r_added)
    r_removed_s.append(r_removed)
    r_difference_s.append(r_difference)








    # q_r_specific_wsls_all.append(quants)
    # b_r_specific_wsls_all.append(bools_s)
    # e_r_specific_wsls_all.append(equals)
    # f_r_specific_wsls_all.append(feats)
    # r_r_specific_wsls_all.append(reals)



  rev_probs_tr.append(np.mean(rev_probs_wsls_all))
  # q_r_specific_wsls.append(q_r_specific_wsls_all)
  # b_r_specific_wsls.append(b_r_specific_wsls_all)
  # e_r_specific_wsls.append(e_r_specific_wsls)
  # f_r_specific_wsls.append(f_r_specific_wsls)
  # r_r_specific_wsls.append(r_r_specific_wsls)

  q_shared_all_tr.append(np.mean(q_shared_s))
  # print(q_shared_all_tr)
  q_difference_all_tr.append(np.mean(q_difference_s))
  q_added_all_tr.append(np.mean(q_added_s))
  q_removed_all_tr.append(np.mean(q_removed_s))


  b_shared_all_tr.append(np.mean(b_shared_s))
  b_difference_all_tr.append(np.mean(b_difference_s))
  b_added_all_tr.append(np.mean(b_added_s))
  b_removed_all_tr.append(np.mean(b_removed_s))


  e_shared_all_tr.append(np.mean(e_shared_s))
  e_difference_all_tr.append(np.mean(e_difference_s))
  e_added_all_tr.append(np.mean(e_added_s))
  e_removed_all_tr.append(np.mean(e_removed_s))


  f_shared_all_tr.append(np.mean(f_shared_s))
  f_difference_all_tr.append(np.mean(f_difference_s))
  f_added_all_tr.append(np.mean(f_added_s))
  f_removed_all_tr.append(np.mean(f_removed_s))

  r_shared_all_tr.append(np.mean(r_shared_s))
  r_difference_all_tr.append(np.mean(r_difference_s))
  r_added_all_tr.append(np.mean(r_added_s))
  r_removed_all_tr.append(np.mean(r_removed_s))

  # print('step')




############### ts learner
# now for process modles
# getting exact rule components
q_r_specific_wsls = []
b_r_specific_wsls = []
e_r_specific_wsls = []
f_r_specific_wsls = []
r_r_specific_wsls = []

q_added_all_ts = []
q_shared_all_ts = []
q_removed_all_ts = []
q_difference_all_ts = []

b_added_all_ts = []
b_shared_all_ts = []
b_removed_all_ts = []
b_difference_all_ts = []

e_added_all_ts = []
e_shared_all_ts = []
e_removed_all_ts = []
e_difference_all_ts = []

f_added_all_ts = []
f_shared_all_ts = []
f_removed_all_ts = []
f_difference_all_ts = []

r_added_all_ts = []
r_shared_all_ts = []
r_removed_all_ts = []
r_difference_all_ts = []

init_probs_ts = init_probs
rev_probs_ts = []
wsls_rules = []

with open('model_results/test4/rules_post_map_chain_clean_1kbest.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()
    for line in filecontents:
        # remove linebreak which is the last character of the string
        current_place = line[:-1]
        # add item to the list
        wsls_rules.append(current_place)


for wsls_rule_list,init_resp in zip(wsls_rules[:248],main_data_formatted['prior_resp'][:248]):
  rev_probs_wsls_all = []
  q_r_specific_wsls_all = []
  b_r_specific_wsls_all = []
  e_r_specific_wsls_all = []
  f_r_specific_wsls_all = []
  r_r_specific_wsls_all = []

  to_list_i = list(rr.get_prec_recursively(rr.string_to_list(init_resp))['to'])
  quants_i = [i for i in to_list_i if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
  bools_i = [i for i in to_list_i if i in ['J(B,B)', 'not_operator(B)']]
  equals_i = [i for i in to_list_i if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
  feats_i = [i for i in to_list_i if i in ['colour','size','orientation','grounded']]
  reals_i = [i for i in to_list_i if i in ['contact']]

  q_added_s = []
  q_shared_s = []
  q_removed_s = []
  q_difference_s = []

  b_added_s = []
  b_shared_s = []
  b_removed_s = []
  b_difference_s = []

  e_added_s = []
  e_shared_s = []
  e_removed_s = []
  e_difference_s = []

  f_added_s = []
  f_shared_s = []
  f_removed_s = []
  f_difference_s = []

  r_added_s = []
  r_shared_s = []
  r_removed_s = []
  r_difference_s = []

  for wsls_rule in eval(wsls_rule_list):
    rev_probs_wsls_all.append(np.log(np.prod(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['li'])))
    from_list = list(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['from'])
    to_list = list(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['to'])



    quants_r = [i for i in to_list if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
    bools_r = [i for i in to_list if i in ['J(B,B)', 'not_operator(B)']]
    equals_r = [i for i in to_list if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
    feats_r = [i for i in to_list if i in ['colour','size','orientation','grounded']]
    reals_r = [i for i in to_list if i in ['contact']]

    # if wsls_rules != init_resp:

    q_shared = sum([a==b for a,b in zip(quants_i,quants_r) if len(quants_i)==len(quants_r)])
    q_added=0
    q_removed=0
    if len(quants_i) > len(quants_r):
      q_removed = len(quants_i) - len(quants_r)
    elif len(quants_i) < len(quants_r):
      q_added = len(quants_r) - len(quants_i)
    q_difference = sum([a!=b for a,b in zip(quants_i,quants_r) if len(quants_i)==len(quants_r)])

    q_shared_s.append(q_shared)
    q_added_s.append(q_added)
    q_removed_s.append(q_removed)
    q_difference_s.append(q_difference)

    b_shared = sum([a==b for a,b in zip(bools_i,bools_r) if len(bools_i)==len(bools_r)])
    b_added=0
    b_removed=0
    if len(bools_i) > len(bools_r):
      b_removed = len(bools_i) - len(bools_r)
    elif len(bools_i) < len(bools_r):
      b_added = len(bools_r) - len(bools_i)
    b_difference = sum([a!=b for a,b in zip(bools_i,bools_r) if len(bools_i)==len(bools_r)])

    b_shared_s.append(b_shared)
    b_added_s.append(b_added)
    b_removed_s.append(b_removed)
    b_difference_s.append(b_difference)



    e_shared = sum([a==b for a,b in zip(equals_i,equals_r) if len(equals_i)==len(equals_r)])
    e_added=0
    e_removed=0
    if len(equals_i) > len(equals_r):
      e_removed = len(equals_i) - len(equals_r)
    elif len(equals_i) < len(equals_r):
      e_added = len(equals_r) - len(equals_i)
    e_difference = sum([a!=b for a,b in zip(equals_i,equals_r) if len(equals_i)==len(equals_r)])

    e_shared_s.append(e_shared)
    e_added_s.append(e_added)
    e_removed_s.append(e_removed)
    e_difference_s.append(e_difference)


    f_shared = sum([a==b for a,b in zip(feats_i,feats_r) if len(feats_i)==len(feats_r)])
    f_added=0
    f_removed=0
    if len(feats_i) > len(feats_r):
      f_removed = len(feats_i) - len(feats_r)
    elif len(feats_i) < len(feats_r):
      f_added = len(feats_r) - len(feats_i)
    f_difference = sum([a!=b for a,b in zip(feats_i,feats_r) if len(feats_i)==len(feats_r)])

    f_shared_s.append(f_shared)
    f_added_s.append(f_added)
    f_removed_s.append(f_removed)
    f_difference_s.append(f_difference)


    r_shared = sum([a==b for a,b in zip(reals_i,reals_r) if len(reals_i)==len(reals_r)])
    r_added=0
    r_removed=0
    if len(reals_i) > len(reals_r):
      r_removed = len(reals_i) - len(reals_r)
    elif len(reals_i) < len(reals_r):
      r_added = len(reals_r) - len(reals_i)
    r_difference = sum([a!=b for a,b in zip(reals_i,reals_r) if len(reals_i)==len(reals_r)])

    r_shared_s.append(r_shared)
    r_added_s.append(r_added)
    r_removed_s.append(r_removed)
    r_difference_s.append(r_difference)








    # q_r_specific_wsls_all.append(quants)
    # b_r_specific_wsls_all.append(bools_s)
    # e_r_specific_wsls_all.append(equals)
    # f_r_specific_wsls_all.append(feats)
    # r_r_specific_wsls_all.append(reals)



  rev_probs_ts.append(np.mean(rev_probs_wsls_all))
  # q_r_specific_wsls.append(q_r_specific_wsls_all)
  # b_r_specific_wsls.append(b_r_specific_wsls_all)
  # e_r_specific_wsls.append(e_r_specific_wsls)
  # f_r_specific_wsls.append(f_r_specific_wsls)
  # r_r_specific_wsls.append(r_r_specific_wsls)

  q_shared_all_ts.append(np.mean(q_shared_s))
  q_difference_all_ts.append(np.mean(q_difference_s))
  q_added_all_ts.append(np.mean(q_added_s))
  q_removed_all_ts.append(np.mean(q_removed_s))


  b_shared_all_ts.append(np.mean(b_shared_s))
  b_difference_all_ts.append(np.mean(b_difference_s))
  b_added_all_ts.append(np.mean(b_added_s))
  b_removed_all_ts.append(np.mean(b_removed_s))


  e_shared_all_ts.append(np.mean(e_shared_s))
  e_difference_all_ts.append(np.mean(e_difference_s))
  e_added_all_ts.append(np.mean(e_added_s))
  e_removed_all_ts.append(np.mean(e_removed_s))


  f_shared_all_ts.append(np.mean(f_shared_s))
  f_difference_all_ts.append(np.mean(f_difference_s))
  f_added_all_ts.append(np.mean(f_added_s))
  f_removed_all_ts.append(np.mean(f_removed_s))

  r_shared_all_ts.append(np.mean(r_shared_s))
  r_difference_all_ts.append(np.mean(r_difference_s))
  r_added_all_ts.append(np.mean(r_added_s))
  r_removed_all_ts.append(np.mean(r_removed_s))



############### normative learner

# getting exact rule components
q_r_specific_wsls = []
b_r_specific_wsls = []
e_r_specific_wsls = []
f_r_specific_wsls = []
r_r_specific_wsls = []

q_added_all_tn = []
q_shared_all_tn = []
q_removed_all_tn = []
q_difference_all_tn = []

b_added_all_tn = []
b_shared_all_tn = []
b_removed_all_tn = []
b_difference_all_tn = []

e_added_all_tn = []
e_shared_all_tn = []
e_removed_all_tn = []
e_difference_all_tn = []

f_added_all_tn = []
f_shared_all_tn = []
f_removed_all_tn = []
f_difference_all_tn = []

r_added_all_tn = []
r_shared_all_tn = []
r_removed_all_tn = []
r_difference_all_tn = []

init_probs_tn = init_probs
rev_probs_tn = []
wsls_rules = []

with open('model_results/norm_1/rules_post_all.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()
    for line in filecontents:
        # remove linebreak which is the last character of the string
        current_place = line[:-1]
        # add item to the list
        wsls_rules.append(current_place)


for wsls_rule_list,init_resp in zip(wsls_rules[:248],main_data_formatted['prior_resp'][:248]):
  rev_probs_wsls_all = []
  q_r_specific_wsls_all = []
  b_r_specific_wsls_all = []
  e_r_specific_wsls_all = []
  f_r_specific_wsls_all = []
  r_r_specific_wsls_all = []

  to_list_i = list(rr.get_prec_recursively(rr.string_to_list(init_resp))['to'])
  quants_i = [i for i in to_list_i if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
  bools_i = [i for i in to_list_i if i in ['J(B,B)', 'not_operator(B)']]
  equals_i = [i for i in to_list_i if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
  feats_i = [i for i in to_list_i if i in ['colour','size','orientation','grounded']]
  reals_i = [i for i in to_list_i if i in ['contact']]

  q_added_s = []
  q_shared_s = []
  q_removed_s = []
  q_difference_s = []

  b_added_s = []
  b_shared_s = []
  b_removed_s = []
  b_difference_s = []

  e_added_s = []
  e_shared_s = []
  e_removed_s = []
  e_difference_s = []

  f_added_s = []
  f_shared_s = []
  f_removed_s = []
  f_difference_s = []

  r_added_s = []
  r_shared_s = []
  r_removed_s = []
  r_difference_s = []

  for wsls_rule in eval(wsls_rule_list):
    rev_probs_wsls_all.append(np.log(np.prod(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['li'])))
    from_list = list(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['from'])
    to_list = list(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['to'])



    quants_r = [i for i in to_list if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
    bools_r = [i for i in to_list if i in ['J(B,B)', 'not_operator(B)']]
    equals_r = [i for i in to_list if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
    feats_r = [i for i in to_list if i in ['colour','size','orientation','grounded']]
    reals_r = [i for i in to_list if i in ['contact']]

    q_shared = sum([a==b for a,b in zip(quants_i,quants_r) if len(quants_i)==len(quants_r)])
    q_added=0
    q_removed=0
    if len(quants_i) > len(quants_r):
      q_removed = len(quants_i) - len(quants_r)
    elif len(quants_i) < len(quants_r):
      q_added = len(quants_r) - len(quants_i)
    q_difference = sum([a!=b for a,b in zip(quants_i,quants_r) if len(quants_i)==len(quants_r)])

    q_shared_s.append(q_shared)
    q_added_s.append(q_added)
    q_removed_s.append(q_removed)
    q_difference_s.append(q_difference)

    b_shared = sum([a==b for a,b in zip(bools_i,bools_r) if len(bools_i)==len(bools_r)])
    b_added=0
    b_removed=0
    if len(bools_i) > len(bools_r):
      b_removed = len(bools_i) - len(bools_r)
    elif len(bools_i) < len(bools_r):
      b_added = len(bools_r) - len(bools_i)
    b_difference = sum([a!=b for a,b in zip(bools_i,bools_r) if len(bools_i)==len(bools_r)])

    b_shared_s.append(b_shared)
    b_added_s.append(b_added)
    b_removed_s.append(b_removed)
    b_difference_s.append(b_difference)



    e_shared = sum([a==b for a,b in zip(equals_i,equals_r) if len(equals_i)==len(equals_r)])
    e_added=0
    e_removed=0
    if len(equals_i) > len(equals_r):
      e_removed = len(equals_i) - len(equals_r)
    elif len(equals_i) < len(equals_r):
      e_added = len(equals_r) - len(equals_i)
    e_difference = sum([a!=b for a,b in zip(equals_i,equals_r) if len(equals_i)==len(equals_r)])

    e_shared_s.append(e_shared)
    e_added_s.append(e_added)
    e_removed_s.append(e_removed)
    e_difference_s.append(e_difference)


    f_shared = sum([a==b for a,b in zip(feats_i,feats_r) if len(feats_i)==len(feats_r)])
    f_added=0
    f_removed=0
    if len(feats_i) > len(feats_r):
      f_removed = len(feats_i) - len(feats_r)
    elif len(feats_i) < len(feats_r):
      f_added = len(feats_r) - len(feats_i)
    f_difference = sum([a!=b for a,b in zip(feats_i,feats_r) if len(feats_i)==len(feats_r)])

    f_shared_s.append(f_shared)
    f_added_s.append(f_added)
    f_removed_s.append(f_removed)
    f_difference_s.append(f_difference)


    r_shared = sum([a==b for a,b in zip(reals_i,reals_r) if len(reals_i)==len(reals_r)])
    r_added=0
    r_removed=0
    if len(reals_i) > len(reals_r):
      r_removed = len(reals_i) - len(reals_r)
    elif len(reals_i) < len(reals_r):
      r_added = len(reals_r) - len(reals_i)
    r_difference = sum([a!=b for a,b in zip(reals_i,reals_r) if len(reals_i)==len(reals_r)])

    r_shared_s.append(r_shared)
    r_added_s.append(r_added)
    r_removed_s.append(r_removed)
    r_difference_s.append(r_difference)








    # q_r_specific_wsls_all.append(quants)
    # b_r_specific_wsls_all.append(bools_s)
    # e_r_specific_wsls_all.append(equals)
    # f_r_specific_wsls_all.append(feats)
    # r_r_specific_wsls_all.append(reals)



  rev_probs_tn.append(np.mean(rev_probs_wsls_all))
  # q_r_specific_wsls.append(q_r_specific_wsls_all)
  # b_r_specific_wsls.append(b_r_specific_wsls_all)
  # e_r_specific_wsls.append(e_r_specific_wsls)
  # f_r_specific_wsls.append(f_r_specific_wsls)
  # r_r_specific_wsls.append(r_r_specific_wsls)

  q_shared_all_tn.append(np.mean(q_shared_s))
  q_difference_all_tn.append(np.mean(q_difference_s))
  q_added_all_tn.append(np.mean(q_added_s))
  q_removed_all_tn.append(np.mean(q_removed_s))


  b_shared_all_tn.append(np.mean(b_shared_s))
  b_difference_all_tn.append(np.mean(b_difference_s))
  b_added_all_tn.append(np.mean(b_added_s))
  b_removed_all_tn.append(np.mean(b_removed_s))


  e_shared_all_tn.append(np.mean(e_shared_s))
  e_difference_all_tn.append(np.mean(e_difference_s))
  e_added_all_tn.append(np.mean(e_added_s))
  e_removed_all_tn.append(np.mean(e_removed_s))


  f_shared_all_tn.append(np.mean(f_shared_s))
  f_difference_all_tn.append(np.mean(f_difference_s))
  f_added_all_tn.append(np.mean(f_added_s))
  f_removed_all_tn.append(np.mean(f_removed_s))

  r_shared_all_tn.append(np.mean(r_shared_s))
  r_difference_all_tn.append(np.mean(r_difference_s))
  r_added_all_tn.append(np.mean(r_added_s))
  r_removed_all_tn.append(np.mean(r_removed_s))



############### normative learner prrior

# getting exact rule components
q_r_specific_wsls = []
b_r_specific_wsls = []
e_r_specific_wsls = []
f_r_specific_wsls = []
r_r_specific_wsls = []

q_added_all_tnp = []
q_shared_all_tnp = []
q_removed_all_tnp = []
q_difference_all_tnp = []

b_added_all_tnp = []
b_shared_all_tnp = []
b_removed_all_tnp = []
b_difference_all_tnp = []

e_added_all_tnp = []
e_shared_all_tnp = []
e_removed_all_tnp = []
e_difference_all_tnp = []

f_added_all_tnp = []
f_shared_all_tnp = []
f_removed_all_tnp = []
f_difference_all_tnp = []

r_added_all_tnp = []
r_shared_all_tnp = []
r_removed_all_tnp = []
r_difference_all_tnp = []

init_probs_tnp = []
rev_probs_tnp = []
wsls_rules_post = copy.deepcopy(wsls_rules)
wsls_rules = []

with open('model_results/norm_1/rules_prior.txt', 'r') as filehandle:
    filecontents = filehandle.readlines()
    for line in filecontents:
        # remove linebreak which is the last character of the string
        current_place = line[:-1]
        # add item to the list
        wsls_rules.append(current_place)


for wsls_rule_list,rev_resp in zip(wsls_rules[:248],wsls_rules_post[:248]):
  rev_probs_wsls_all = []
  q_r_specific_wsls_all = []
  b_r_specific_wsls_all = []
  e_r_specific_wsls_all = []
  f_r_specific_wsls_all = []
  r_r_specific_wsls_all = []

  q_added_s = []
  q_shared_s = []
  q_removed_s = []
  q_difference_s = []

  b_added_s = []
  b_shared_s = []
  b_removed_s = []
  b_difference_s = []

  e_added_s = []
  e_shared_s = []
  e_removed_s = []
  e_difference_s = []

  f_added_s = []
  f_shared_s = []
  f_removed_s = []
  f_difference_s = []

  r_added_s = []
  r_shared_s = []
  r_removed_s = []
  r_difference_s = []

  for wsls_rule,rev_response in zip(eval(wsls_rule_list),eval(rev_resp)):
    rev_probs_wsls_all.append(np.log(np.prod(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['li'])))
    from_list_i = list(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['from'])
    to_list_i = list(rr.get_prec_recursively(rr.string_to_list(wsls_rule))['to'])
    to_list = list(rr.get_prec_recursively(rr.string_to_list(rev_response))['to'])

    quants_i = [i for i in to_list_i if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
    bools_i = [i for i in to_list_i if i in ['J(B,B)', 'not_operator(B)']]
    equals_i = [i for i in to_list_i if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
    feats_i = [i for i in to_list_i if i in ['colour','size','orientation','grounded']]
    reals_i = [i for i in to_list_i if i in ['contact']]


    quants_r = [i for i in to_list if i in ['Z.exactly','Z.atleast','Z.atmost','Z.exists','Z.forall']]
    bools_r = [i for i in to_list if i in ['J(B,B)', 'not_operator(B)']]
    equals_r = [i for i in to_list if i in ['Z.hor_operator(xN,xO,I)','Z.equal(xN,D)','Z.equal(xN,xO,G)','Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']]
    feats_r = [i for i in to_list if i in ['colour','size','orientation','grounded']]
    reals_r = [i for i in to_list if i in ['contact']]

    q_shared = sum([a==b for a,b in zip(quants_i,quants_r) if len(quants_i)==len(quants_r)])
    q_added=0
    q_removed=0
    if len(quants_i) > len(quants_r):
      q_removed = len(quants_i) - len(quants_r)
    elif len(quants_i) < len(quants_r):
      q_added = len(quants_r) - len(quants_i)
    q_difference = sum([a!=b for a,b in zip(quants_i,quants_r) if len(quants_i)==len(quants_r)])

    q_shared_s.append(q_shared)
    q_added_s.append(q_added)
    q_removed_s.append(q_removed)
    q_difference_s.append(q_difference)

    b_shared = sum([a==b for a,b in zip(bools_i,bools_r) if len(bools_i)==len(bools_r)])
    b_added=0
    b_removed=0
    if len(bools_i) > len(bools_r):
      b_removed = len(bools_i) - len(bools_r)
    elif len(bools_i) < len(bools_r):
      b_added = len(bools_r) - len(bools_i)
    b_difference = sum([a!=b for a,b in zip(bools_i,bools_r) if len(bools_i)==len(bools_r)])

    b_shared_s.append(b_shared)
    b_added_s.append(b_added)
    b_removed_s.append(b_removed)
    b_difference_s.append(b_difference)



    e_shared = sum([a==b for a,b in zip(equals_i,equals_r) if len(equals_i)==len(equals_r)])
    e_added=0
    e_removed=0
    if len(equals_i) > len(equals_r):
      e_removed = len(equals_i) - len(equals_r)
    elif len(equals_i) < len(equals_r):
      e_added = len(equals_r) - len(equals_i)
    e_difference = sum([a!=b for a,b in zip(equals_i,equals_r) if len(equals_i)==len(equals_r)])

    e_shared_s.append(e_shared)
    e_added_s.append(e_added)
    e_removed_s.append(e_removed)
    e_difference_s.append(e_difference)


    f_shared = sum([a==b for a,b in zip(feats_i,feats_r) if len(feats_i)==len(feats_r)])
    f_added=0
    f_removed=0
    if len(feats_i) > len(feats_r):
      f_removed = len(feats_i) - len(feats_r)
    elif len(feats_i) < len(feats_r):
      f_added = len(feats_r) - len(feats_i)
    f_difference = sum([a!=b for a,b in zip(feats_i,feats_r) if len(feats_i)==len(feats_r)])

    f_shared_s.append(f_shared)
    f_added_s.append(f_added)
    f_removed_s.append(f_removed)
    f_difference_s.append(f_difference)


    r_shared = sum([a==b for a,b in zip(reals_i,reals_r) if len(reals_i)==len(reals_r)])
    r_added=0
    r_removed=0
    if len(reals_i) > len(reals_r):
      r_removed = len(reals_i) - len(reals_r)
    elif len(reals_i) < len(reals_r):
      r_added = len(reals_r) - len(reals_i)
    r_difference = sum([a!=b for a,b in zip(reals_i,reals_r) if len(reals_i)==len(reals_r)])

    r_shared_s.append(r_shared)
    r_added_s.append(r_added)
    r_removed_s.append(r_removed)
    r_difference_s.append(r_difference)








    # q_r_specific_wsls_all.append(quants)
    # b_r_specific_wsls_all.append(bools_s)
    # e_r_specific_wsls_all.append(equals)
    # f_r_specific_wsls_all.append(feats)
    # r_r_specific_wsls_all.append(reals)



  rev_probs_tnp.append(np.mean(rev_probs_wsls_all))
  # q_r_specific_wsls.append(q_r_specific_wsls_all)
  # b_r_specific_wsls.append(b_r_specific_wsls_all)
  # e_r_specific_wsls.append(e_r_specific_wsls)
  # f_r_specific_wsls.append(f_r_specific_wsls)
  # r_r_specific_wsls.append(r_r_specific_wsls)

  q_shared_all_tnp.append(np.mean(q_shared_s))
  q_difference_all_tnp.append(np.mean(q_difference_s))
  q_added_all_tnp.append(np.mean(q_added_s))
  q_removed_all_tnp.append(np.mean(q_removed_s))


  b_shared_all_tnp.append(np.mean(b_shared_s))
  b_difference_all_tnp.append(np.mean(b_difference_s))
  b_added_all_tnp.append(np.mean(b_added_s))
  b_removed_all_tnp.append(np.mean(b_removed_s))


  e_shared_all_tnp.append(np.mean(e_shared_s))
  e_difference_all_tnp.append(np.mean(e_difference_s))
  e_added_all_tnp.append(np.mean(e_added_s))
  e_removed_all_tnp.append(np.mean(e_removed_s))


  f_shared_all_tnp.append(np.mean(f_shared_s))
  f_difference_all_tnp.append(np.mean(f_difference_s))
  f_added_all_tnp.append(np.mean(f_added_s))
  f_removed_all_tnp.append(np.mean(f_removed_s))

  r_shared_all_tnp.append(np.mean(r_shared_s))
  r_difference_all_tnp.append(np.mean(r_difference_s))
  r_added_all_tnp.append(np.mean(r_added_s))
  r_removed_all_tnp.append(np.mean(r_removed_s))





rule_comps = pd.DataFrame({'q_i_s': q_i_specific, 'b_i_s': b_i_specific, 'e_i_s': e_i_specific, 'f_i_s': f_i_specific, 'r_i_s': r_i_specific,'init_prob': init_probs,
                           'q_r_s': q_r_specific, 'b_r_s': b_r_specific, 'e_r_s': e_r_specific, 'f_r_s': f_r_specific, 'r_r_s': r_r_specific, 'rev_probs': rev_probs,
                           'q_wsls_shared': q_shared_all, 'q_wsls_difference': q_difference_all, 'q_wsls_added': q_added_all, 'q_wsls_removed': q_removed_all,
                           'e_wsls_shared': e_shared_all, 'e_wsls_difference': e_difference_all, 'e_wsls_added': e_added_all, 'e_wsls_removed': e_removed_all,
                           'f_wsls_shared': f_shared_all, 'f_wsls_difference': f_difference_all, 'f_wsls_added': f_added_all, 'f_wsls_removed': f_removed_all,
                           'r_wsls_shared': r_shared_all, 'r_wsls_difference': r_difference_all, 'r_wsls_added': r_added_all, 'r_wsls_removed': r_removed_all,
                           'b_wsls_shared': b_shared_all, 'b_wsls_difference': b_difference_all, 'b_wsls_added': b_added_all, 'b_wsls_removed': b_removed_all,'rev_probs_wsls':rev_probs_wsls,

                           'q_tr_shared': q_shared_all_tr, 'q_tr_difference': q_difference_all_tr, 'q_tr_added': q_added_all_tr, 'q_tr_removed': q_removed_all_tr,
                           'e_tr_shared': e_shared_all_tr, 'e_tr_difference': e_difference_all_tr, 'e_tr_added': e_added_all_tr, 'e_tr_removed': e_removed_all_tr,
                           'f_tr_shared': f_shared_all_tr, 'f_tr_difference': f_difference_all_tr, 'f_tr_added': f_added_all_tr, 'f_tr_removed': f_removed_all_tr,
                           'r_tr_shared': r_shared_all_tr, 'r_tr_difference': r_difference_all_tr, 'r_tr_added': r_added_all_tr, 'r_tr_removed': r_removed_all_tr,
                           'b_tr_shared': b_shared_all_tr, 'b_tr_difference': b_difference_all_tr, 'b_tr_added': b_added_all_tr, 'b_tr_removed': b_removed_all_tr,'rev_probs_tr': rev_probs_tr,


                           'q_ts_shared': q_shared_all_ts, 'q_ts_difference': q_difference_all_ts, 'q_ts_added': q_added_all_ts, 'q_ts_removed': q_removed_all_ts,
                           'e_ts_shared': e_shared_all_ts, 'e_ts_difference': e_difference_all_ts, 'e_ts_added': e_added_all_ts, 'e_ts_removed': e_removed_all_ts,
                           'f_ts_shared': f_shared_all_ts, 'f_ts_difference': f_difference_all_ts, 'f_ts_added': f_added_all_ts, 'f_ts_removed': f_removed_all_ts,
                           'r_ts_shared': r_shared_all_ts, 'r_ts_difference': r_difference_all_ts, 'r_ts_added': r_added_all_ts, 'r_ts_removed': r_removed_all_ts,
                           'b_ts_shared': b_shared_all_ts, 'b_ts_difference': b_difference_all_ts, 'b_ts_added': b_added_all_ts, 'b_ts_removed': b_removed_all_ts,'rev_probs_ts': rev_probs_ts,


                           'q_tn_shared': q_shared_all_tn, 'q_tn_difference': q_difference_all_tn, 'q_tn_added': q_added_all_tn, 'q_tn_removed': q_removed_all_tn,
                           'e_tn_shared': e_shared_all_tn, 'e_tn_difference': e_difference_all_tn, 'e_tn_added': e_added_all_tn, 'e_tn_removed': e_removed_all_tn,
                           'f_tn_shared': f_shared_all_tn, 'f_tn_difference': f_difference_all_tn, 'f_tn_added': f_added_all_tn, 'f_tn_removed': f_removed_all_tn,
                           'r_tn_shared': r_shared_all_tn, 'r_tn_difference': r_difference_all_tn, 'r_tn_added': r_added_all_tn, 'r_tn_removed': r_removed_all_tn,
                           'b_tn_shared': b_shared_all_tn, 'b_tn_difference': b_difference_all_tn, 'b_tn_added': b_added_all_tn, 'b_tn_removed': b_removed_all_tn,'rev_probs_tn': rev_probs_tn,
                           # then process models

                           'q_tnp_shared': q_shared_all_tnp, 'q_tnp_difference': q_difference_all_tnp, 'q_tnp_added': q_added_all_tnp, 'q_tnp_removed': q_removed_all_tnp,
                           'e_tnp_shared': e_shared_all_tnp, 'e_tnp_difference': e_difference_all_tnp, 'e_tnp_added': e_added_all_tnp, 'e_tnp_removed': e_removed_all_tnp,
                           'f_tnp_shared': f_shared_all_tnp, 'f_tnp_difference': f_difference_all_tnp, 'f_tnp_added': f_added_all_tnp, 'f_tnp_removed': f_removed_all_tnp,
                           'r_tnp_shared': r_shared_all_tnp, 'r_tnp_difference': r_difference_all_tnp, 'r_tnp_added': r_added_all_tnp, 'r_tnp_removed': r_removed_all_tnp,
                           'b_tnp_shared': b_shared_all_tnp, 'b_tnp_difference': b_difference_all_tnp, 'b_tnp_added': b_added_all_tnp, 'b_tnp_removed': b_removed_all_tnp,'rev_probs_tnp': rev_probs_tnp
                           })
# print('hiheloos')
rule_comps.to_csv('model_results/rule_comp_cond_1_1k_best.csv')
