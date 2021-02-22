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
from scipy.optimize import minimize
# print(pd.__version__)  # needs to be a relatively recent version (older versions coming with python <= 3.6 do not work)



####################### Custom imports ########################################
from mcmc_cond_1_new import mcmc_sampler, mcmc_sampler_map_surgery, mcmc_sampler_map
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
main_data_formatted = pd.read_csv('main_data_formatted_cond_three_cut_rules_second_rule.csv')  # getting the preprocessed data file


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
all_are_the_same_size = "ob.forall(lambda x1: ob.forall(lambda x2: ob.equal(x1,x2,'size'), X), X)"
contact = "ob.exists(lambda x1: ob.exists(lambda x2: ob.hor_operator(x1,x2,'contact'), X), X)"
blue_to_red_contact = "ob.exists(lambda x1: ob.exists(lambda x2: ob.and_operator(ob.and_operator(ob.equal(x1, 'blue','colour'), ob.equal(x2 , 'red', 'colour')), ob.hor_operator(x1,x2,'contact')), X), X)"
red_bigger_than_all_nonred = "ob.exists(lambda x1: ob.forall(lambda x2: ob.or_operator(ob.and_operator(ob.equal(x1,'red','colour'), ob.greater(x1,x2,'size')), ob.equal(x2, 'red', 'colour')), X), X)"
stacked = "ob.exists(lambda x1: ob.exists(lambda x2: ob.and_operator(ob.and_operator(ob.and_operator(ob.and_operator(ob.and_operator(ob.equal(x1,'upright','orientation'),ob.equal(x1,'yes','grounded')),ob.equal(x2,'upright','orientation')),ob.equal(x2,'no','grounded')),ob.equal(x1,x2,'xpos')),ob.hor_operator(x1,x2,'contact')),X),X)"

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

####################### Sampling Algorithm #########################################
def predicted_selections(main_data_formatted, rules_dict,  replacements, trial_counts, n_rep = 1, n_1=1, n_2=5):  # computes the ll for getting participants responses to initial generalizations (n_1 * n_2 determines number of MCMC iterations)
     rep = 0                            # index for number of repititions of the whole sampling procedure

     for repeat in np.arange(n_rep):     # for each repeat, an independent outuput file will be created
          i = 0                          # index over trials (= n_subjects * n_trials per subject)
          gt = [1,1,1,1,0,0,0,0]         # ground truth for generalisations (first four are always correct, last four always wrong)

          # computing additional variables for a single trial for subjects prior responses
          fitted_taus_prior = []        # fitted temperature parameters for each trial for each participant (dictates soft vs hard maximisation)
          raw_probs_prior = []          # raw probabilities for each scene (i.e. if they follow a rule or not)
          select_probs_prior = []       # selection probabilities for each scene based on tau
          ll_model_prior = []           # negative log likelihood of observing participant data for a given rule (i.e. sum of all log ll for each scene for one rule)
          BICs_model_prior = []         # BIC
          ll_baseline_prior = []        # same for basline
          BICs_baseline_prior = []


          # computing the same variables for all 5 trials for each subject (just aggregating over trials for later model comparison)
          raw_probs_all_trials_prior = []
          raw_probs_all_trials_one_list_prior = []
          select_probs_all_trials_prior = []
          prior_resp_all_trials_prior = []
          fitted_taus_all_trials_prior = []
          ll_model_all_trials_prior = []
          BICs_model_all_trials_prior = []
          ll_baseline_all_trials_prior = []
          BICs_baseline_all_trials_prior = []

          # repeating above for prior labels predicting posterior labels
          fitted_taus_prior_labels = []        # fitted temperature parameters for each trial for each participant (dictates soft vs hard maximisation)
          raw_probs_prior_labels = []          # raw probabilities for each scene (i.e. if they follow a rule or not)
          select_probs_prior_labels = []       # selection probabilities for each scene based on tau
          ll_model_prior_labels = []           # negative log likelihood of observing participant data for a given rule (i.e. sum of all log ll for each scene for one rule)
          BICs_model_prior_labels = []         # BIC


          # computing the same variables for all 5 trials for each subject (just aggregating over trials for later model comparison)
          raw_probs_all_trials_prior_labels = []
          raw_probs_all_trials_one_list_prior_labels = []
          select_probs_all_trials_prior_labels = []
          prior_resp_all_trials_prior_labels = []
          fitted_taus_all_trials_prior_labels = []
          ll_model_all_trials_prior_labels = []
          BICs_model_all_trials_prior_labels = []


          # repeating the above for subjects posteriors based on map estimates

          fitted_taus_post_map = []        # fitted temperature parameters for each trial for each participant (dictates soft vs hard maximisation)
          raw_probs_post_map = []          # raw probabilities for each scene (i.e. if they follow a rule or not)
          select_probs_post_map = []       # selection probabilities for each scene based on tau
          ll_model_post_map = []           # negative log likelihood of observing participant data for a given rule (i.e. sum of all log ll for each scene for one rule)
          BICs_model_post_map = []         # BIC
          ll_baseline_post_map = []        # same for basline
          BICs_baseline_post_map = []


          raw_probs_all_trials_post_map = []
          raw_probs_all_trials_one_list_post_map = []
          select_probs_all_trials_post_map = []
          post_resp_all_trials_post_map = []
          fitted_taus_all_trials_post_map = []
          ll_model_all_trials_post_map = []
          BICs_model_all_trials_post_map = []
          ll_baseline_all_trials_post_map = []
          BICs_baseline_all_trials_post_map = []

          # # repeating the above for subjects posteriores based on all 16 data points
          fitted_taus_post_all = []        # fitted temperature parameters for each trial for each participant (dictates soft vs hard maximisation)
          raw_probs_post_all = []          # raw probabilities for each scene (i.e. if they follow a rule or not)
          select_probs_post_all = []       # selection probabilities for each scene based on tau
          ll_model_post_all = []           # negative log likelihood of observing participant data for a given rule (i.e. sum of all log ll for each scene for one rule)
          BICs_model_post_all = []         # BIC
          ll_baseline_post_all = []        # same for basline
          BICs_baseline_post_all = []


          raw_probs_all_trials_post_all = []
          raw_probs_all_trials_one_list_post_all = []
          select_probs_all_trials_post_all = []
          post_resp_all_trials_post_all = []
          fitted_taus_all_trials_post_all = []
          ll_model_all_trials_post_all = []
          BICs_model_all_trials_post_all = []
          ll_baseline_all_trials_post_all = []
          BICs_baseline_all_trials_post_all = []

          fitted_taus_post_all_seed = []        # fitted temperature parameters for each trial for each participant (dictates soft vs hard maximisation)
          raw_probs_post_all_seed = []          # raw probabilities for each scene (i.e. if they follow a rule or not)
          select_probs_post_all_seed = []       # selection probabilities for each scene based on tau
          ll_model_post_all_seed = []           # negative log likelihood of observing participant data for a given rule (i.e. sum of all log ll for each scene for one rule)
          BICs_model_post_all_seed = []         # BIC
          ll_baseline_post_all_seed = []        # same for basline
          BICs_baseline_post_all_seed = []


          raw_probs_all_trials_post_all_seed = []
          raw_probs_all_trials_one_list_post_all_seed= []
          select_probs_all_trials_post_all_seed = []
          post_resp_all_trials_post_all_seed = []
          fitted_taus_all_trials_post_all_seed = []
          ll_model_all_trials_post_all_seed = []
          BICs_model_all_trials_post_all_seed = []
          ll_baseline_all_trials_post_all_seed = []
          BICs_baseline_all_trials_post_all_seed = []


          # accuracy of different models
          prior_accs = []
          prior_accs_single = []
          prior_label_accs = []
          prior_label_accs_single = []
          post_map_accs = []
          post_map_accs_single = []
          post_all_accs = []
          post_all_accs_single = []
          post_all_accs_seed = []
          post_all_accs_single_seed = []


          # labels (= 1 if model is better fit for subject and 0 if baseline is better fit for subject)
          prior_labels = []
          prior_labels_labels = []
          post_map_labels = []
          post_all_labels = []
          post_all_labels_seed = []

          rules_prior = []
          rules_prior_label = []
          rules_post_map = []
          rules_post_all = []
          rules_post_all_seed = []

          # maps of each sampline procedure (= the hypothesis that occured most often)
          map_prior = []
          map_prior_label = []
          map_post_map = []
          map_post_all = []
          map_post_all_seed = []

          # accuracy of maps
          map_prior_acc = []
          map_prior_acc_label = []
          map_post_map_acc = []
          map_post_all_acc = []
          map_post_all_acc_seed = []


          # count how often correct rule occurs
          correct_rule_perc_prior = []
          correct_rule_perc_prior_labels = []
          correct_rule_perc_post_map = []
          correct_rule_perc_post_all = []
          correct_rule_perc_post_all_seed = []


          n_rows =82
          n_trials_counter = 1

          #  looping over each trial for each subject (n_trials * n_subjects iterations and run mcmc chains)
          for data in main_data_formatted['data'][:n_rows]:

               Dwin =  [.25,.25,0,0,0,.25,.25]

               # getting name and id to create a unique csv file for each mcmc chain for each subject and trial
               rule_name = main_data_formatted['rule_name'][i]
               # print(rule_name)
               rule_string = rules_dict[rule_name]
               # print(rule_string)

               subj_rule = main_data_formatted['prior_resp'][i]
               subj_rev_rule = main_data_formatted['post_resp'][i]
               bv_prior = eval(main_data_formatted['bound_vars'][i])
               # print(subj_rule)
               # print(bv_prior)

               correct_rule = rules_dict[rule_name]
               token_id = main_data_formatted['token_id'][i]
               n_trials = trial_counts[token_id]

               # getting subjects prior and posterior responses
               prior_response = eval(main_data_formatted['prior'][i])
               posterior_response = eval(main_data_formatted['posterior'][i])

               # getting the data
               init_trials = eval(data)[:8]           # trials = scenes created by participants
               rev_trials =  eval(main_data_formatted['data'][i])[:8]
               generalizations = eval(data)[8:]
               full_data = init_trials + rev_trials # full data used for the third mcmc chain combining trials and generalizations


               # getting additional training data for evaluation of model
               training_data = rand_scene_creator(correct_rule, n_scenes=0)
               full_training_dat_prior = init_trials + training_data
               label_dat_post_map = generalizations
               full_training_dat_post_map = rev_trials + label_dat_post_map
               full_training_dat_post_all = full_data + training_data


               prior_probs = get_production_probs_prototype(full_training_dat_prior,'prior',cond='1',feat_only=False)
               Dwin_prior = prior_probs[0]
               feat_probs_prior = prior_probs[1]


               prob_gen_follow_rule_prior_label = []

               map_res_prior_label = []


               # evaluating the rules based on the generalization data shown to subjects
               for gen in generalizations:        # looping over all 8 generalization scenes



                    res_prior_label = []




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
#                     # print(X)
#                     # print(ob.exists(lambda x1: ob.forall(lambda x2: ob.and_operator(ob.and_operator(ob.equal(x1,'red','colour'), ob.not_operator(ob.equal(x2, 'red', 'colour'))), ob.greater(x1,x2,'size')), X), X))
                    # evaluating all sampled rules against the scenes for each of the different mcmc chains and appending results
                    res_prior_label.append(eval(subj_rev_rule))
#                          # print(eval(rule))
#                     # print(X)
#                     # print(ob.forall(lambda x1: ob.not_operator(ob.equal(x1, 'upright', 'orientation')), X))
                    # for rule in df_prior_labels['rulestring']:
#                     #      print(subj_rule)


                    map_res_prior_label.append(eval(subj_rev_rule))


                    # computing the raw probabilities that the scenes follow a rule for each chain

                    p_follow_rule_prior_label = (1 / len(res_prior_label)) * sum(res_prior_label) # len(res) = number of rules; sum(res) = number of rules matching the scene
#                     # print(p_follow_rule_prior)
                    print(len(res_prior_label))
                    print(sum(res_prior_label))
                    prob_gen_follow_rule_prior_label.append(p_follow_rule_prior_label)




               map_prior_acc_label.append(compute_acc(gt, map_res_prior_label))

               # return 1
               # fitting tau to the data using a generic minimize function from scipy.optimize for all chains


                # prior chain
               raw_probs_prior_labels.append(prob_gen_follow_rule_prior_label)  # only used for single trial examples
               raw_probs_all_trials_prior_labels.append(prob_gen_follow_rule_prior_label)
               prior_resp_all_trials_prior_labels.append(posterior_response)
               fitted_tau_prior_label = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(prob_gen_follow_rule_prior_label,posterior_response), method='L-BFGS-B')
               fitted_taus_prior_labels.append(fitted_tau_prior_label.x[0])
               fitted_results_mod_prior_label = fitted_probs(fitted_tau_prior_label.x[0], prob_gen_follow_rule_prior_label, posterior_response)
               select_probs_prior_labels.append(fitted_results_mod_prior_label[0])
               ll_model_prior_labels.append(fitted_results_mod_prior_label[1])
               BICs_model_prior_labels.append(-2 * fitted_results_mod_prior_label[1] + 1 * np.log(8))





               #
               ####################### ALL TRIALS FOR ONE SUBJECT ##########################################
               # print(len(fitted_taus_prior))
               # print(n_trials)
               # print('ntrialsabove')
               if n_trials_counter % n_trials == 0:
                    n_trials_counter = 0
                    # print(n_trials_counter)

                    raw_probs_all_trials_label_1 = [prob for sublist in raw_probs_all_trials_prior_labels[i-n_trials+1:] for prob in sublist]
                    prior_resp_all_trials_label_1 = [response for sublist in prior_resp_all_trials_prior_labels[i-n_trials+1:] for response in sublist]


                    overall_fitted_tau_label = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_trials_label_1,prior_resp_all_trials_label_1), method='L-BFGS-B')
                    fitted_results_all_trials_mod_label = fitted_probs(overall_fitted_tau_label.x[0], raw_probs_all_trials_label_1, prior_resp_all_trials_label_1)





                    # print(len(fitted_taus_prior))
                    # print('fittedtausprior')
                    if len(fitted_taus_prior) % len(main_data_formatted['rule_name'][:n_rows]) == 0:
                         # print('vat')
                         # len(main_data_formatted['rule_name'][:n_rows])

                         # prior
#                          # print('potato')
                         raw_probs_all_subjects = [prob for sublist in raw_probs_all_trials_prior for prob in sublist]
                         print(raw_probs_all_subjects)
                         prior_resp_all_subjects = [response for sublist in prior_resp_all_trials_prior for response in sublist]

                         # labels
                         raw_probs_all_subjects_label = [prob for sublist in raw_probs_all_trials_prior_labels for prob in sublist]

                         prior_resp_all_subjects_label = [response for sublist in prior_resp_all_trials_prior_labels for response in sublist]

                         fitted_tau_all_subjects_prior_label = minimize(get_tau, 1, bounds=[(0.1, 100.00)],args=(raw_probs_all_subjects_label,prior_resp_all_subjects_label), method='L-BFGS-B')
                         print(fitted_tau_all_subjects_prior_label)
                         print(raw_probs_all_subjects_label)
                         print(prior_resp_all_subjects_label)
                         fitted_results_all_subjects_prior_label = fitted_probs(fitted_tau_all_subjects_prior_label.x[0], raw_probs_all_subjects_label,prior_resp_all_subjects_label)


                         fitted_taus_all_subjects_prior_label = []
                         fitted_taus_all_subjects_prior_label.append(float(fitted_tau_all_subjects_prior_label.x[0]))

                         select_probs_all_subjects_prior_label = []
                         select_probs_all_subjects_prior_label.append(fitted_results_all_subjects_prior_label[0])

                         ll_model_all_subjects_prior_label = []
                         ll_model_all_subjects_prior_label.append(fitted_results_all_subjects_prior_label[1])

                         BICs_model_all_subjects_prior_label = []
                         BICs_model_all_subjects_prior_label.append(-2 * fitted_results_all_subjects_prior_label[1] + 1 * np.log(8 * len(main_data_formatted['rule_name'][:n_rows])))

                         # post map

                         gt_all_subjects = gt * len(fitted_taus_prior) # ground truth


                         prior_mod_select_all_subj_label = hard_max_selections(raw_probs_all_subjects_label)

                         prior_acc_all_subj_label = sum([a and b or not a and not b for a, b in zip(gt_all_subjects, prior_mod_select_all_subj_label)]) / len(prior_mod_select_all_subj_label)
                           # print(len(fitted_taus_prior))








                    for trial in range(n_trials):



                         fitted_taus_all_trials_prior_labels.append(overall_fitted_tau_label.x[0])
                         select_probs_all_trials_prior_labels.append(fitted_results_all_trials_mod_label[0])
                         ll_model_all_trials_prior_labels.append(fitted_results_all_trials_mod_label[1])
                         BICs_model_all_trials_prior_labels.append(-2 * fitted_results_all_trials_mod_label[1] + 1 * np.log(8*n_trials))
                         raw_probs_all_trials_one_list_prior_labels.append(raw_probs_all_trials_prior_labels)





                    # computing accuracy of model predictions using hard maximization for selection probs
                    gt_all = gt * n_trials # ground truth


                    prior_mod_select_labels = hard_max_selections(raw_probs_all_trials_label_1)



                    prior_acc_label = sum([a and b or not a and not b for a, b in zip(gt_all, prior_mod_select_labels)]) / len(prior_mod_select_labels)

                    low_bound = 0
                    up_bound = 8
                    for acc in range(n_trials):

                         prior_label_accs.append(prior_acc_label)


                         prior_label_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, prior_mod_select_labels[low_bound:up_bound])]) / 8)

                         low_bound+=8
                         up_bound+=8







               i+=1 # pr
               n_trials_counter+=1
               print(i)
               # oceeding to next trial


          ################ APPENDING ALL DATA TO MAIN DATA FRAME ########################


#           # print(raw_probs_all_trials_one_list_prior)
          # prior trial specific data

          main_data_formatted = main_data_formatted[:n_rows]


          # main_data_formatted['select_probs_fitted_tau_prior'] = select_probs_prior

          #now labels
          # main_data_formatted['raw_probs_prior'] = raw_probs_prior
          main_data_formatted['fitted_tau_prior_label'] = fitted_taus_prior_labels
          main_data_formatted['log_ll_model_prior_label'] = ll_model_prior_labels
          main_data_formatted['BIC_model_prior_label'] = BICs_model_prior_labels

          # main_data_formatted['select_probs_fitted_tau_prior'] = select_probs_prior

          # prior data across trials
          # main_data_formatted['raw_probs_all_trials_prior'] = raw_probs_all_trials_one_list_prior
          main_data_formatted['fitted_tau_all_trials_prior_label'] = fitted_taus_all_trials_prior_labels
          main_data_formatted['log_ll_model_all_trials_prior_label'] = ll_model_all_trials_prior_labels
          main_data_formatted['BIC_model_all_trials_prior_label'] = BICs_model_all_trials_prior_labels

          # main_data_formatted['select_probs_fitted_tau_all_trials_prior'] = select_probs_all_trials_prior

          # print('giantpotato')
          # # print(fitted_taus_all_subjects_prior)
          # # print(len(select_probs_all_subjects_prior))
          # # print(select_probs_all_subjects_prior)
          # # print(len(select_probs_all_subjects_prior[0]))
          # # print(ll_model_all_subjects_prior)
          # # print(BICs_model_all_subjects_prior)
          # # print(ll_baseline_all_subjects_prior)
          # # print(BICs_baseline_all_subjects_prior)
          main_data_formatted['fitted_taus_all_subjects_prior_label'] = fitted_taus_all_subjects_prior_label * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['raw_probs_all_subjects_prior_label'] = [raw_probs_all_subjects_label] * len(main_data_formatted['rule_name'][:n_rows])
#           # print(len(select_probs_all_subjects_prior * 450)
          main_data_formatted['ll_model_all_subjects_prior_label'] = ll_model_all_subjects_prior_label* len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['BICs_model_all_subjects_prior_label'] = BICs_model_all_subjects_prior_label* len(main_data_formatted['rule_name'][:n_rows])
          # main_data_formatted['ll_baseline_all_subjects_prior'] = ll_baseline_all_subjects_prior* len(main_data_formatted['rule_name'][:n_rows])
          # main_data_formatted['BICs_baseline_all_subjects_prior'] = BICs_baseline_all_subjects_prior* len(main_data_formatted['rule_name'][:n_rows])




          main_data_formatted['prior_label_acc'] = prior_label_accs
          main_data_formatted['prior_acc_single_label'] = prior_label_accs_single



          main_data_formatted['all_sub_prior_acc_label'] = prior_acc_all_subj_label




          main_data_formatted.to_csv('model_results/normative_res_three_rev_rule_only.csv')   # writing main data to new csv file including all relevant data for analysis

          rep+=1

predicted_selections(main_data_formatted, rules_dict, replacements, trial_counts, n_rep=1)
