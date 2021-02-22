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
from scipy.optimize import minimize
# print(pd.__version__)  # needs to be a relatively recent version (older versions coming with python <= 3.6 do not work)



####################### Custom imports ########################################
from mcmc_cond_1_new_test_change_seq_poiss_3 import mcmc_sampler, mcmc_sampler_map_surgery, mcmc_sampler_map
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
main_data_formatted = pd.read_csv('../data/main_data_cond_one_old.csv')  # getting the preprocessed data file
print(len(main_data_formatted['data']))

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
#
main_data_formatted = main_data_formatted.query("post_resp != 's'")
main_data_formatted = main_data_formatted.reset_index(drop=True)
# # removing all complex rules from the data frame, allowing only the five simple booleans to remain (Zeta, Upsilon, Iota, Kappa, Omega)
main_data_formatted = main_data_formatted.query("rule_name == 'Zeta' or rule_name == 'Upsilon' or rule_name == 'Iota' or rule_name == 'Kappa' or rule_name == 'Omega'")
main_data_formatted = main_data_formatted.reset_index(drop=True)

# print(len(main_data_formatted['rule_name']))  # remaining number of trials (450/450)

# creating dictionary with subject's tokens as keys and the number of trials each subject completed after removing complex rules as values
trial_counts = main_data_formatted.groupby('token_id').size()

trial_counts = dict(trial_counts)

print(trial_counts)
print(sum(list(trial_counts.values())))

####################### Sampling Algorithm #########################################
def predicted_selections(main_data_formatted, rules_dict,  replacements, trial_counts, n_rep = 1, n_1=1, n_2=5):  # computes the ll for getting participants responses to initial generalizations (n_1 * n_2 determines number of MCMC iterations)
     rep = 0                            # index for number of repititions of the whole sampling procedure

     for repeat in np.arange(n_rep):     # for each repeat, an independent outuput file will be created
          i = 0                          # index over trials (= n_subjects * n_trials per subject)
          gt = [1,1,1,1,0,0,0,0]         # ground truth for generalisations (first four are always correct, last four always wrong)
          global X


          n_rows =248
          n_trials_counter = 1
          init_count = []
          rev_count = []

          init_count_mod = []
          rev_count_mod = []

          corr_init_mod  = []
          corr_rev_mod = []
          corr_init_subj = []
          corr_rev_subj = []


          with open('model_results/norm_2/rules_prior_205.txt', 'r') as filehandle:
              filecontents = filehandle.readlines()
              for line in filecontents:
                  current_place = line[:-1]
                  init_count_mod.append(eval(current_place))


          with open('model_results/norm_2/rules_post_all_450.txt', 'r') as filehandle:
              filecontents = filehandle.readlines()
              for line in filecontents:
                  current_place = line[:-1]
                  rev_count_mod.append(eval(current_place))

          #  looping over each trial for each subject (n_trials * n_subjects iterations and run mcmc chains)

          corr_pred_inti_subj = []
          corr_pred_rev_subj = []
          same_subj = []

          for data in main_data_formatted['data'][:n_rows]:

               rule_name = main_data_formatted['rule_name'][i]
               correct_rule = rules_dict[rule_name]

               test_data = []

               with open('../data/1000_random_subj_scenes.txt', 'r') as filehandle:
                   filecontents = filehandle.readlines()
                   for line in filecontents:
                       current_place = line[:-1]
                       test_data.append(eval(current_place))

               test_data = random.sample(test_data,100)
               print(len(test_data))





               mod_count_step_init = []
               mod_count_step_rev = []

               # mod_init = init_count_mod[i]
               # mod_rev = rev_count_mod[i]



               corr_pred_init = []
               corr_pred_rev = []



               # for rule_init in mod_init:
               #
               #     res_truth = []
               #     res_mod_init_truth = []
               #     for i_6 in range(0, len(test_data)):
               #         X = []
               #         for i_7 in range(0, len(test_data[i_6]['ids'])):  # looping over the number of objects in each scene
               #             contact = check_structure(test_data[i_6]['contact'], i_7)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
               #          # getting the properties for each object (triangle / cone) in the scene
               #             object = {"id": test_data[i_6]['ids'][i_7], "colour":  test_data[i_6]['colours'][i_7] , "size":  test_data[i_6]['sizes'][i_7], "xpos":  int(np.round(test_data[i_6]['xpos'][i_7])),
               #                 "ypos": int(np.round(test_data[i_6]['ypos'][i_7])), "rotation":  np.round(test_data[i_6]['rotations'][i_7],1), "orientation":  test_data[i_6]['orientations'][i_7],
               #                 "grounded":  test_data[i_6]['grounded'][i_7], "contact":  contact}
               #             X.append(object)   # appending the object to X which includes all objects for a scene
               #      # res_t.append(eval(t["rule"]))             # evaluating the rules against X
               #         res_mod_init_truth.append(eval(rule_init))
               #         res_truth.append(eval(correct_rule))
               #     if res_mod_init_truth == res_truth:
               #
               #         corr_pred_init.append(1)
               #     else:
               #         corr_pred_init.append(0)
               #
               #
               # i_6=0
               # i_7=0
               # for rule_rev in mod_rev:
               #
               #     res_truth = []
               #     res_mod_rev_truth = []
               #
               #     for i_6 in range(0, len(test_data)):
               #         X = []
               #         for i_7 in range(0, len(test_data[i_6]['ids'])):  # looping over the number of objects in each scene
               #             contact = check_structure(test_data[i_6]['contact'], i_7)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
               #          # getting the properties for each object (triangle / cone) in the scene
               #             object = {"id": test_data[i_6]['ids'][i_7], "colour":  test_data[i_6]['colours'][i_7] , "size":  test_data[i_6]['sizes'][i_7], "xpos":  int(np.round(test_data[i_6]['xpos'][i_7])),
               #             "ypos": int(np.round(test_data[i_6]['ypos'][i_7])), "rotation":  np.round(test_data[i_6]['rotations'][i_7],1), "orientation":  test_data[i_6]['orientations'][i_7],
               #             "grounded":  test_data[i_6]['grounded'][i_7], "contact":  contact}
               #             X.append(object)   # appending the object to X which includes all objects for a scene
               #      # res_t.append(eval(t["rule"]))             # evaluating the rules against X
               #         res_mod_rev_truth.append(eval(rule_rev))
               #         res_truth.append(eval(correct_rule))
               #     if res_mod_rev_truth == res_truth:
               #         corr_pred_rev.append(1)
               #
               #     else:
               #         corr_pred_rev.append(0)


               # mod_count_step_init = sum(corr_pred_init ) / len(corr_pred_init)
               # mod_count_step_rev = sum(corr_pred_rev) / len(corr_pred_rev)
               # print(len(corr_pred_init))
               #
               # corr_init_mod.append(mod_count_step_init)
               # corr_rev_mod.append(mod_count_step_rev)




               subj_rule_init = main_data_formatted['prior_resp'][i]
               subj_rule_init = rr.list_to_string(rr.string_to_list(subj_rule_init))
               subj_rule_rev = main_data_formatted['post_resp'][i]
               subj_rule_rev = rr.list_to_string(rr.string_to_list(subj_rule_rev))

               i_6 = 0
               i_7 = 0

               res_truth = []
               res_subj_init_truth = []
               for i_6 in range(0, len(test_data)):
                   X = []
                   for i_7 in range(0, len(test_data[i_6]['ids'])):  # looping over the number of objects in each scene
                       contact = check_structure(test_data[i_6]['contact'], i_7)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
                    # getting the properties for each object (triangle / cone) in the scene
                       object = {"id": test_data[i_6]['ids'][i_7], "colour":  test_data[i_6]['colours'][i_7] , "size":  test_data[i_6]['sizes'][i_7], "xpos":  int(np.round(test_data[i_6]['xpos'][i_7])),
                           "ypos": int(np.round(test_data[i_6]['ypos'][i_7])), "rotation":  np.round(test_data[i_6]['rotations'][i_7],1), "orientation":  test_data[i_6]['orientations'][i_7],
                           "grounded":  test_data[i_6]['grounded'][i_7], "contact":  contact}
                       X.append(object)   # appending the object to X which includes all objects for a scene
                # res_t.append(eval(t["rule"]))             # evaluating the rules against X
                   res_subj_init_truth.append(eval(subj_rule_init))
                   res_truth.append(eval(subj_rule_rev))
               if res_subj_init_truth == res_truth:
                   corr_pred_inti_subj.append(1)
               else:
                   corr_pred_inti_subj.append(0)


               # i_6=0
               # i_7=0
               #
               # res_truth = []
               # res_subj_rev_truth = []
               # for i_6 in range(0, len(test_data)):
               #     X = []
               #     for i_7 in range(0, len(test_data[i_6]['ids'])):  # looping over the number of objects in each scene
               #         contact = check_structure(test_data[i_6]['contact'], i_7)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
               #      # getting the properties for each object (triangle / cone) in the scene
               #         object = {"id": test_data[i_6]['ids'][i_7], "colour":  test_data[i_6]['colours'][i_7] , "size":  test_data[i_6]['sizes'][i_7], "xpos":  int(np.round(test_data[i_6]['xpos'][i_7])),
               #             "ypos": int(np.round(test_data[i_6]['ypos'][i_7])), "rotation":  np.round(test_data[i_6]['rotations'][i_7],1), "orientation":  test_data[i_6]['orientations'][i_7],
               #             "grounded":  test_data[i_6]['grounded'][i_7], "contact":  contact}
               #         X.append(object)   # appending the object to X which includes all objects for a scene
               #  # res_t.append(eval(t["rule"]))             # evaluating the rules against X
               #
               #     res_subj_rev_truth.append(eval(subj_rule_rev))
               #     res_truth.append(eval(correct_rule))
               # if res_subj_rev_truth == res_truth:
               #     corr_pred_rev_subj.append(1)
               # else:
               #
               #     corr_pred_rev_subj.append(0)

               # res_same_init = []
               # res_same_rev = []
               # for i_6 in range(0, len(test_data)):
               #     X = []
               #     for i_7 in range(0, len(test_data[i_6]['ids'])):  # looping over the number of objects in each scene
               #         contact = check_structure(test_data[i_6]['contact'], i_7)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
               #      # getting the properties for each object (triangle / cone) in the scene
               #         object = {"id": test_data[i_6]['ids'][i_7], "colour":  test_data[i_6]['colours'][i_7] , "size":  test_data[i_6]['sizes'][i_7], "xpos":  int(np.round(test_data[i_6]['xpos'][i_7])),
               #             "ypos": int(np.round(test_data[i_6]['ypos'][i_7])), "rotation":  np.round(test_data[i_6]['rotations'][i_7],1), "orientation":  test_data[i_6]['orientations'][i_7],
               #             "grounded":  test_data[i_6]['grounded'][i_7], "contact":  contact}
               #         X.append(object)   # appending the object to X which includes all objects for a scene
               #  # res_t.append(eval(t["rule"]))             # evaluating the rules against X
               #
               #     res_same_rev.append(eval(subj_rule_rev))
               #     res_same_init.append(eval(subj_rule_init))
               # if res_same_init == res_same_rev:
               #     same_subj.append(1)
               # else:
               #     same_subj.append(0)
               #
               #
               #
               #
               #




               print(same_subj)
               n_trials_counter+=1
               print(corr_pred_init)
               print(corr_init_mod)
               i+=1
          print(same_subj)
          # #
          main_data_formatted['same_rule_subject'] = corr_pred_inti_subj
          # main_data_formatted['corr_rule_subj_revised'] = corr_pred_rev_subj
          # main_data_formatted['corr_rule_mod_initial'] = corr_init_mod
          # main_data_formatted['corr_rule_mod_revised'] = corr_rev_mod
          main_data_formatted.to_csv('exp3_same_rule_subj_82.csv')

          # with open('exp_1_corr_rule_subj.txt', 'w') as filehandle:
          #     filehandle.writelines("%s\n" % place for place in same_subj)
          # # with open('model_results/test4/rules_post_all_chain_clean.txt', 'w') as filehandle:
          # #     filehandle.writelines("%s\n" % place for place in rules_post_all)
          # with open('model_results/exp_1_corr_rule_subj_revised.txt', 'w') as filehandle:
          #     filehandle.writelines("%s\n" % place for place in rev_count)
          #
          # with open('model_results/exp_2_corr_rule_model_initial_210.txt', 'w') as filehandle:
          #     filehandle.writelines("%s\n" % place for place in corr_init_mod)
          # # with open('model_results/test4/rules_post_all_chain_clean.txt', 'w') as filehandle:
          # #     filehandle.writelines("%s\n" % place for place in rules_post_all)
          # with open('model_results/exp_2_corr_rule_model_revised_210.txt', 'w') as filehandle:
          #     filehandle.writelines("%s\n" % place for place in corr_rev_mod)



          # rep+=1

predicted_selections(main_data_formatted, rules_dict, replacements, trial_counts, n_rep=1)
