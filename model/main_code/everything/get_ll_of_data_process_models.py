"""
Created on Sun Mar 15 15:01:21 2020

@author: jan-philippfranken
"""
###############################################################################
########################### Main File #########################################
###############################################################################


####################### General imports #######################################
import numpy as np
from statistics import mode
import pandas as pd
import random
from formal_learning_model import likelihood
from scipy.optimize import minimize
# print(pd.__version__)  # needs to be a relatively recent version (older versions coming with python <= 3.6 do not work)



####################### Custom imports ########################################
# from mcmc_cond_1_new_test_change import mcmc_sampler, mcmc_sampler_map_surgery, mcmc_sampler_map
from pcfg_generator import pcfg_generator
from tree_regrower import tree_regrower
from transform_functions import compute_orientation, check_structure, get_production_probs_prototype
from tau_ml_estimation import get_tau, fitted_probs, compare_BIC, hard_max_selections, compute_acc, compute_distance
from create_random_scenes import rand_scene_creator
from reverse_rule import reverse_rule



###################### Preliminaries ##########################################
Z = pcfg_generator()                                         # instantiating pcfg generator
tg = tree_regrower()
rr = reverse_rule()
random.seed(1)                                                # setting random.seed to allow replication of mcmc chain
main_data_formatted = pd.read_csv('main_data_formatted_cond_two_second_rule.csv')  # getting the preprocessed data file


####################### grammar ##############################################
S = ['Z.exists(lambda xN: A,X)', 'Z.forall(lambda xN: A,X)', 'L(lambda xN: A,M,X)']
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

# replacement dictionary
replacements = {"S": ["S"],
                "A": ['Z.exists','Z.forall','Z.atleast','Z.atmost','Z.exactly'],
                "B": ['Z.equal', 'Z.hor_operator', 'Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater','Z.and_operator','Z.or_operator','Z.not_operator'],
                "C": ['Z.equal', 'Z.hor_operator', 'Z.lequal', 'Z.grequal', 'Z.less', 'Z.greater']}


Z = pcfg_generator()  # instantiating grammar generator (Z is an arbitrary choice, the letter G is already used in the grammar...)
tg = tree_regrower()  # instantiating tree regrower


############################# rules used in the experiment ##################################
# simple booleans
there_is_a_red = "Z.exists(lambda x1: Z.equal(x1,'red','colour'),X)"
nothing_is_upright = "Z.forall(lambda x1: Z.not_operator(Z.equal(x1,'upright','orientation')),X)"
one_is_blue = "Z.exactly(lambda x1: Z.equal(x1,'blue','colour'),1,X)"
there_is_a_blue_and_small = "Z.exists(lambda x1: Z.and_operator(Z.equal(x1,1,'size'),Z.equal(x1,'blue','colour')),X)"
all_are_blue_or_small = "Z.forall(lambda x1: Z.or_operator(Z.equal(x1,1,'size'),Z.equal(x1,'blue','colour')),X)"


# more complex rules (not relvant for the present comparison focusing only on simple booleans)
all_are_the_same_size = "Z.forall(lambda x1: Z.forall(lambda x2: Z.equal(x1,x2,'size'), X), X)"
contact = "Z.exists(lambda x1: Z.exists(lambda x2: Z.hor_operator(x1,x2,'contact'), X), X)"
blue_to_red_contact = "Z.exists(lambda x1: Z.exists(lambda x2: Z.and_operator(Z.and_operator(Z.equal(x1, 'blue','colour'), Z.equal(x2 , 'red', 'colour')), Z.hor_operator(x1,x2,'contact')), X), X)"
red_bigger_than_all_nonred = "Z.exists(lambda x1: Z.forall(lambda x2: Z.or_operator(Z.and_operator(Z.equal(x1,'red','colour'), Z.greater(x1,x2,'size')), Z.equal(x2, 'red', 'colour')), X), X)"
stacked = "Z.exists(lambda x1: Z.exists(lambda x2: Z.and_operator(Z.and_operator(Z.and_operator(Z.and_operator(Z.and_operator(Z.equal(x1,'upright','orientation'),Z.equal(x1,'yes','grounded')),Z.equal(x2,'upright','orientation')),Z.equal(x2,'no','grounded')),Z.equal(x1,x2,'xpos')),Z.hor_operator(x1,x2,'contact')),X),X)"

# summarising rules in dictionary
rules_dict = { # simple booleans
              'Zeta': there_is_a_red,
              'Upsilon': nothing_is_upright,
              'Iota': one_is_blue,
              'Kappa': there_is_a_blue_and_small,
              'Omega': all_are_blue_or_small,
              # complex rules
              'Phi': all_are_the_same_size,
              'Nu': contact,
              'Xi': blue_to_red_contact,
              'Mu': red_bigger_than_all_nonred,
              'Psi': stacked}

main_data_formatted = main_data_formatted.query("post_resp != 's'")
main_data_formatted = main_data_formatted.reset_index(drop=True)
# removing all complex rules from the data frame, allowing only the five simple booleans to remain (Zeta, Upsilon, Iota, Kappa, Omega)
main_data_formatted = main_data_formatted.query("rule_name == 'Zeta' or rule_name == 'Upsilon' or rule_name == 'Iota' or rule_name == 'Kappa' or rule_name == 'Omega'")
main_data_formatted = main_data_formatted.reset_index(drop=True)

# print(len(main_data_formatted['rule_name']))  # remaining number of trials (450/450)

# creating dictionary with subject's tokens as keys and the number of trials each subject completed after removing complex rules as values
trial_counts = main_data_formatted.groupby('token_id').size()

trial_counts = dict(trial_counts)

print(trial_counts)
print(sum(list(trial_counts.values())))


post_rules_wsls = []
post_rules_tr = []
post_rules_ts = []

with open('model_results/test5/rules_post_all_chain_clean.txt', 'r') as filehandle:
   filecontents = filehandle.readlines()
   for line in filecontents:
    # remove linebreak which is the last character of the string
        current_place = line[:-1]
    # add item to the list
        post_rules_wsls.append(eval(current_place))

with open('model_results/test5/rules_post_all_seed_chain_clean_qual_lambda3.txt', 'r') as filehandle:
   filecontents = filehandle.readlines()
   for line in filecontents:
    # remove linebreak which is the last character of the string
        current_place = line[:-1]
    # add item to the list
        post_rules_tr.append(eval(current_place))


with open('model_results/test5/rules_post_map_chain_clean_qual_lambda3.txt', 'r') as filehandle:
   filecontents = filehandle.readlines()
   for line in filecontents:
    # remove linebreak which is the last character of the string
        current_place = line[:-1]
    # add item to the list
        post_rules_ts.append(eval(current_place))

####################### Sampling Algorithm #########################################
def predicted_selections(main_data_formatted, rules_dict,  replacements, trial_counts, n_rep = 1, n_1=1, n_2=5):  # computes the ll for getting participants responses to initial generalizations (n_1 * n_2 determines number of MCMC iterations)
     rep = 0                            # index for number of repititions of the whole sampling procedure

     for repeat in np.arange(n_rep):     # for each repeat, an independent outuput file will be created
          i = 0                          # index over trials (= n_subjects * n_trials per subject)

          n_rows =248
          n_trials_counter = 1
          average_ll_wsls = []
          average_ll_tr = []
          average_ll_ts = []

          #  looping over each trial for each subject (n_trials * n_subjects iterations and run mcmc chains)
          for data in main_data_formatted['partner_data'][:n_rows]:

               Dwin =  [.25,.25,0,0,0,.25,.25]

               # getting name and id to create a unique csv file for each mcmc chain for each subject and trial
               rule_name = main_data_formatted['rule_name'][i]
               # print(rule_name)
               rule_string = rules_dict[rule_name]
               # print(rule_string)

               subj_rule = main_data_formatted['prior_resp'][i]
               bv_prior = eval(main_data_formatted['bound_vars'][i])
               subj_rule = rr.list_to_string(rr.string_to_list(subj_rule))
               print(rule_string)
               print(bv_prior)

               correct_rule = rules_dict[rule_name]
               token_id = main_data_formatted['token_id'][i]
               n_trials = trial_counts[token_id]

               # getting subjects prior and posterior responses
               prior_response = eval(main_data_formatted['prior'][i])
               posterior_response = eval(main_data_formatted['posterior'][i])

               # getting the data
               init_trials = eval(data)[:8]           # trials = scenes created by participants
               rev_trials =  eval(main_data_formatted['partner_data'][i])[:8]
               generalizations = eval(data)[8:]

               full_data = eval(data)[:8] + eval(main_data_formatted['partner_data'][i])[:8] # full data used for the third mcmc chain combining trials and generalizations



               ll_post_wsls = []
               ll_post_tr = []
               ll_post_ts = []



               # evaluating the rules based on the generalization data shown to subjects
               for rule in post_rules_tr[i]:
                   gen_count = 0
                   ground_truth = []
                   res_post_tr = []                # result checks whether this scene follows a rule (length of results equals n_1 * n_2)

                   for gen in rev_trials:        # looping over all 8 generalization scenes


                        global X                # defining a global variable X
                        X = []                  # for each scene, X will include the objects (i.e., cones) of the scene

                        # looping over the number of objects (ie cones) in each scene
                        for i_3 in range(0, len(gen['ids'])):
    #                          # print(gen['contact'])
                             contact = check_structure(gen['contact'], i_3)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
    #                          # print(contact)
                             # getting the properties for each object (triangle / cone) in the scene
                             object = {"id": gen['ids'][i_3], "colour":  gen['colours'][i_3] , "size":  gen['sizes'][i_3], "xpos":  int(np.round(gen['xpos'][i_3])),
                             "ypos":  int(np.round(gen['ypos'][i_3])), "rotation": np.round(gen['rotations'][i_3],1), "orientation": compute_orientation(gen['rotations'][i_3])[0],
                             "grounded":  gen['grounded'][i_3], "contact":  contact}

                             X.append(object)   # appending the object to X which includes all objects for a scene
                        ground_truth.append(rev_trials[gen_count]['follow_rule'])
                        gen_count+=1
                        res_post_tr.append(eval(rule))

                   z_t = [a and b or not a and not b for a, b in zip(ground_truth, res_post_tr)]  # comparing results with ground truth
                   # print(z_t[0])
                   out_tr = len(z_t) - sum(z_t)  # counts the number of false classifications for a rule (outliers)

                  # 3b) computing the likelihood of each rule under consideration of the number of outliers
                   ll = likelihood()  # see formal learning model for details on likelihood
                   ll_tr = ll.compute_ll(4, out_tr)
                   ll_post_tr.append(ll_tr)




               for rule_2 in post_rules_wsls[i]:
                   gen_2_count = 0

                   ground_truth = []
                   res_post_wsls = []                # result checks whether this scene follows a rule (length of results equals n_1 * n_2)

                   for gen_2 in full_data:        # looping over all 8 generalization scenes
                       #
                       # ground_truth = []
                       # res_post = []                # result checks whether this scene follows a rule (length of results equals n_1 * n_2)

                        # global X                # defining a global variable X
                       X = []                  # for each scene, X will include the objects (i.e., cones) of the scene

                        # looping over the number of objects (ie cones) in each scene

                       for i_4 in range(0, len(gen_2['ids'])):
                           # print(gen_2['contact'])
                           # print(gen_2['ids'])
                           # print(i_4)
                           # print(len(gen_2['ids']))
                           contact = check_structure(gen_2['contact'], i_4)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
#                          # print(contact)
                         # getting the properties for each object (triangle / cone) in the scene
                           object = {"id": gen_2['ids'][i_4], "colour":  gen_2['colours'][i_4] , "size":  gen_2['sizes'][i_4], "xpos":  int(np.round(gen_2['xpos'][i_4])),
                           "ypos":  int(np.round(gen_2['ypos'][i_4])), "rotation": np.round(gen_2['rotations'][i_4],1), "orientation": compute_orientation(gen_2['rotations'][i_4])[0],
                           "grounded":  gen_2['grounded'][i_4], "contact":  contact}

                           X.append(object)   # appending the object to X which includes all objects for a scene
                       res_post_wsls.append(eval(rule_2))
                       ground_truth.append(full_data[gen_2_count]['follow_rule'])
                       gen_2_count+=1




                   z_t = [a and b or not a and not b for a, b in zip(ground_truth, res_post_wsls)]  # comparing results with ground truth

                   out_wsls = len(z_t) - sum(z_t)  # counts the number of false classifications for a rule (outliers)

                  # 3b) computing the likelihood of each rule under consideration of the number of outliers
                   ll_wsls = ll.compute_ll(4, out_wsls)
                   ll_post_wsls.append(ll_wsls)
                   # print(len(ll_post))


               # evaluating the rules based on the generalization data shown to subjects
               for rule_3 in post_rules_ts[i]:
                   gen_count_3 = 0
                   ground_truth = []
                   res_post_ts = []                # result checks whether this scene follows a rule (length of results equals n_1 * n_2)

                   for gen_3 in rev_trials:        # looping over all 8 generalization scenes



                        X = []                  # for each scene, X will include the objects (i.e., cones) of the scene

                        # looping over the number of objects (ie cones) in each scene
                        for i_5 in range(0, len(gen_3['ids'])):
    #                          # print(gen['contact'])
                             contact = check_structure(gen_3['contact'], i_5)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
    #                          # print(contact)
                             # getting the properties for each object (triangle / cone) in the scene
                             object = {"id": gen_3['ids'][i_5], "colour":  gen_3['colours'][i_5] , "size":  gen_3['sizes'][i_5], "xpos":  int(np.round(gen_3['xpos'][i_5])),
                             "ypos":  int(np.round(gen_3['ypos'][i_5])), "rotation": np.round(gen_3['rotations'][i_5],1), "orientation": compute_orientation(gen_3['rotations'][i_5])[0],
                             "grounded":  gen_3['grounded'][i_5], "contact":  contact}

                             X.append(object)   # appending the object to X which includes all objects for a scene
                        ground_truth.append(rev_trials[gen_count_3]['follow_rule'])
                        gen_count_3+=1
                        res_post_ts.append(eval(rule_3))

                   z_t = [a and b or not a and not b for a, b in zip(ground_truth, res_post_ts)]  # comparing results with ground truth
                   # print(z_t[0])
                   out_ts = len(z_t) - sum(z_t)  # counts the number of false classifications for a rule (outliers)

                  # 3b) computing the likelihood of each rule under consideration of the number of outliers
                   ll = likelihood()  # see formal learning model for details on likelihood
                   ll_ts = ll.compute_ll(4, out_ts)
                   ll_post_ts.append(ll_ts)


               average_ll_wsls.append(np.mean(ll_post_wsls))
               average_ll_tr.append(np.mean(ll_post_tr))
               average_ll_ts.append(np.mean(ll_post_ts))
               i+=1
               n_trials_counter+=1
#                # print(i)


          average_ll_wsls = [np.log(item) for item in average_ll_wsls]
          average_ll_tr = [np.log(item) for item in average_ll_tr]
          average_ll_ts = [np.log(item) for item in average_ll_ts]
          with open('model_results/post_wsls_ll_cond_2.txt', 'w') as filehandle:
              filehandle.writelines("%s\n" % place for place in average_ll_wsls)
          with open('model_results/post_tr_ll_lamb3_cond_2.txt', 'w') as filehandle:
              filehandle.writelines("%s\n" % place for place in average_ll_tr)
          with open('model_results/post_ts_ll_lamb3_cond_2.txt', 'w') as filehandle:
              filehandle.writelines("%s\n" % place for place in average_ll_ts)


          rep+=1

predicted_selections(main_data_formatted, rules_dict, replacements, trial_counts, n_rep=1)
