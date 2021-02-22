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

# # print(equalities_i[:248])
# # print(list(main_data_formatted['prior_resp'][:248]))


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
rev_probs_tr_all = []
wsls_rules_all = []

for trial_run in range(0,20):
    print('hello'+str(trial_run))


    rev_probs_tr = []
    wsls_rules = []

    with open('model_results/test4/rules_post_all_seed_chain_clean_10best'+str(trial_run)+'.txt', 'r') as filehandle:
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



      rev_probs_tr_all.append(np.mean(rev_probs_wsls_all))
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
rev_probs_ts_all = []
wsls_rules_all = []

for trial_run in range(0,20):
    print('hello'+str(trial_run))

    rev_probs_ts = []
    wsls_rules = []

    with open('model_results/test4/rules_post_map_chain_clean_10best'+str(trial_run)+'.txt', 'r') as filehandle:
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



      rev_probs_ts_all.append(np.mean(rev_probs_wsls_all))
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

print(q_i_specific)
print(q_shared_all_tr)
print(rev_probs_tr_all)

def comp_mean(input,n):
  output = []
  mean_count = []
  for i in input:
    mean_count.append(i)
    if len(mean_count) == n:
      output.append(np.mean(mean_count))
      mean_count = []
  return output


rule_comps = pd.DataFrame({'q_i_s': q_i_specific, 'b_i_s': b_i_specific, 'e_i_s': e_i_specific, 'f_i_s': f_i_specific, 'r_i_s': r_i_specific,'init_prob': init_probs,
                           'q_r_s': q_r_specific, 'b_r_s': b_r_specific, 'e_r_s': e_r_specific, 'f_r_s': f_r_specific, 'r_r_s': r_r_specific, 'rev_probs': rev_probs,

                           'q_tr_shared': comp_mean(q_shared_all_tr,20), 'q_tr_difference': comp_mean(q_difference_all_tr,20), 'q_tr_added': comp_mean(q_added_all_tr,20), 'q_tr_removed': comp_mean(q_removed_all_tr,20),
                           'e_tr_shared': comp_mean(e_shared_all_tr,20), 'e_tr_difference': comp_mean(e_difference_all_tr,20), 'e_tr_added': comp_mean(e_added_all_tr,20), 'e_tr_removed': comp_mean(e_removed_all_tr,20),
                           'f_tr_shared': comp_mean(f_shared_all_tr,20), 'f_tr_difference': comp_mean(f_difference_all_tr,20), 'f_tr_added': comp_mean(f_added_all_tr,20), 'f_tr_removed': comp_mean(f_removed_all_tr,20),
                           'r_tr_shared': comp_mean(r_shared_all_tr,20), 'r_tr_difference': comp_mean(r_difference_all_tr,20), 'r_tr_added': comp_mean(r_added_all_tr,20), 'r_tr_removed': comp_mean(r_removed_all_tr,20),
                           'b_tr_shared': comp_mean(b_shared_all_tr,20), 'b_tr_difference': comp_mean(b_difference_all_tr,20), 'b_tr_added': comp_mean(b_added_all_tr,20), 'b_tr_removed': comp_mean(b_removed_all_tr,20),'rev_probs_tr': comp_mean(rev_probs_tr_all,20),

                           'q_ts_shared': comp_mean(q_shared_all_ts,20), 'q_ts_difference': comp_mean(q_difference_all_ts,20), 'q_ts_added': comp_mean(q_added_all_ts,20), 'q_ts_removed': comp_mean(q_removed_all_ts,20),
                           'e_ts_shared': comp_mean(e_shared_all_ts,20), 'e_ts_difference': comp_mean(e_difference_all_ts,20), 'e_ts_added': comp_mean(e_added_all_ts,20), 'e_ts_removed': comp_mean(e_removed_all_ts,20),
                           'f_ts_shared': comp_mean(f_shared_all_ts,20), 'f_ts_difference': comp_mean(f_difference_all_ts,20), 'f_ts_added': comp_mean(f_added_all_ts,20), 'f_ts_removed': comp_mean(f_removed_all_ts,20),
                           'r_ts_shared': comp_mean(r_shared_all_ts,20), 'r_ts_difference': comp_mean(r_difference_all_ts,20), 'r_ts_added': comp_mean(r_added_all_ts,20), 'r_ts_removed': comp_mean(r_removed_all_ts,20),
                           'b_ts_shared': comp_mean(b_shared_all_ts,20), 'b_ts_difference': comp_mean(b_difference_all_ts,20), 'b_ts_added': comp_mean(b_added_all_ts,20), 'b_ts_removed': comp_mean(b_removed_all_ts,20),'rev_probs_tr': comp_mean(rev_probs_tr_all,20),

                           })
# print('hiheloos')
rule_comps.to_csv('model_results/rule_comp_cond_1_10_best.csv')
