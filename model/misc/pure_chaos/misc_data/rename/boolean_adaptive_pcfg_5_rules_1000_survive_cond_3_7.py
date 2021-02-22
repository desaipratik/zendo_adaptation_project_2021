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
print(pd.__version__)  # needs to be a relatively recent version (older versions coming with python <= 3.6 do not work)



####################### Custom imports ########################################
from pcfg_generate_rules_subtree_regeneration_restricted_grammar_cond_3 import mh_mcmc_sampler_res
from pruned_pcfg import pcfg_generator
from transform_functions import compute_orientation, check_structure, get_production_probs
from tau_ml_estimation import get_tau, fitted_probs, compare_BIC, hard_max_selections, compute_acc, compute_distance
from create_random_scenes import rand_scene_creator



###################### Preliminaries ##########################################
ob = pcfg_generator()                                         # instantiating pcfg generator
random.seed(1)                                                # setting random.seed to allow replication of mcmc chain
main_data_formatted = pd.read_csv('main_data_formatted_cond_3.csv')  # getting the preprocessed data file


####################### grammar ##############################################
S = ['ob.exists(lambda xN: A, X)', 'ob.forall(lambda xN: A, X)', 'L(lambda xN: A, M, X)']
A = ['B', 'S']
B = ['C', 'J(B,B)', 'ob.not_operator(B)']
C = ['ob.equal(xN, D)', 'K(xN, E)', 'ob.equal(xN, xO, G)', 'K(xN, xO, H)', 'ob.hor_operator(xN,xO,I)']
D = {"colour": ["'red'", "'blue'", "'green'"], "size": [1, 2, 3], "xpos": np.arange(9),
     "ypos": np.arange(2, 6), "rotation": np.arange(0, 6.5, 0.5),
     "orientation": ["'upright'", "'lhs'", "'rhs'", "'strange'"], "grounded": ["'no'", "'yes'"]}
E = {"size": [1, 2, 3], "xpos": np.arange(9), "ypos": np.arange(2, 6), "rotation": np.arange(0, 6.3, 0.1)}
G = ["'colour'", "'size'", "'xpos'", "'ypos'", "'rotation'", "'orientation'", "'grounded'"]
H = ["'size'", "'xpos'", "'ypos'", "'rotation'"]
I = ["'contact'"]
J = ['ob.and_operator', 'ob.or_operator']
K = ['ob.lequal', 'ob.grequal', 'ob.less', 'ob.greater']
L = ['ob.atleast', 'ob.atmost', 'ob.exactly']
M = [1, 2, 3]

# summarizing grammar in dictionary
productions = {"S": S, "A": A, "B": B, "C": C, "D": D, "E": E, "G": G, "H": H, "I": I, "J": J, "K": K, "L": L, "M": M}


##### default production probabilities setting non-booleans and continuous features to 0 ####
Swin = [0.333333333,0.333333333,0.333333333]
Awin = [1,0]
Bwin = [0.33,.33,.33]
Cwin = [.5,.5,0,0,0]
Dwin = [.25,.25,0,0,0,.25,.25]
Ewin = [1,0,0,0]
Gwin = [.25,.25,0,0,0,.25,.25]
Hwin = [1,0,0,0]
Iwin = [0]
Jwin = [.5,.5]
Kwin = [.25,.25,.25,.25]
Lwin = [0.333333333,0.333333333,0.333333333]
Mwin = [1,0,0]


############################# rules used in the experiment ##################################
# simple booleans
there_is_a_red = "ob.exists(lambda x1: ob.equal(x1, 'red','colour'), X)"
nothing_is_upright = "ob.forall(lambda x1: ob.not_operator(ob.equal(x1, 'upright', 'orientation')), X)"
one_is_blue = "ob.exactly(lambda x1: ob.equal(x1, 'blue','colour'), 1, X)"
there_is_a_blue_and_small = "ob.exists(lambda x1: ob.and_operator(ob.equal(x1, 1,'size'),ob.equal(x1, 'blue','colour')), X)"
all_are_blue_or_small = "ob.forall(lambda x1: ob.or_operator(ob.equal(x1, 1,'size'),ob.equal(x1, 'blue','colour')), X)"


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

# removing all complex rules from the data frame, allowing only the five simple booleans to remain (Zeta, Upsilon, Iota, Kappa, Omega)
main_data_formatted = main_data_formatted.query("rule_name == 'Zeta' or rule_name == 'Upsilon' or rule_name == 'Iota' or rule_name == 'Kappa' or rule_name == 'Omega'")
main_data_formatted = main_data_formatted.reset_index(drop=True)
print(len(main_data_formatted['rule_name']))  # remaining number of trials (450/450)

# creating dictionary with subject's tokens as keys and the number of trials each subject completed after removing complex rules as values
trial_counts = main_data_formatted.groupby('token_id').size()
trial_counts = dict(trial_counts)

####################### Sampling Algorithm #########################################
def predicted_selections(main_data_formatted, rules_dict, trial_counts, n_rep = 5, n_1=1, n_2=10000):  # computes the ll for getting participants responses to initial generalizations (n_1 * n_2 determines number of MCMC iterations)
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


          # accuracy of different models
          prior_accs = []
          prior_accs_single = []
          post_map_accs = []
          post_map_accs_single = []
          post_all_accs = []
          post_all_accs_single = []


          # labels (= 1 if model is better fit for subject and 0 if baseline is better fit for subject)
          prior_labels = []
          post_map_labels = []
          post_all_labels = []

          # maps of each sampline procedure (= the hypothesis that occured most often)
          map_prior = []
          map_post_map = []
          map_post_all = []

          # accuracy of maps
          map_prior_acc = []
          map_post_map_acc = []
          map_post_all_acc = []

          #  looping over each trial for each subject (n_trials * n_subjects iterations and run mcmc chains)
          for data in main_data_formatted['data_prior']:


               # getting name and id to create a unique csv file for each mcmc chain for each subject and trial
               rule_name = main_data_formatted['rule_name'][i]
               print(rule_name)

               correct_rule = rules_dict[rule_name]
               token_id = main_data_formatted['token_id'][i]
               n_trials = trial_counts[token_id]



               # getting subjects prior and posterior responses
               prior_response = eval(main_data_formatted['prior'][i])
               posterior_response = eval(main_data_formatted['posterior'][i])

               # getting the data
               init_trials = eval(data)[:8]           # trials = scenes created by participants
               rev_trials =  eval(main_data_formatted['data_posterior'][i])[:8]
               generalizations = eval(data)[8:]
               full_data = init_trials + rev_trials # full data used for the third mcmc chain combining trials and generalizations


               # getting additional training data for evaluation of model
               training_data = rand_scene_creator(correct_rule, n_scenes=0)
               full_training_dat_prior = init_trials + training_data
               full_training_dat_post_map = rev_trials + training_data
               full_training_dat_post_all = full_data + training_data



               Dwin_prior = get_production_probs(full_training_dat_prior,'prior')[0]
               feat_probs_prior = get_production_probs(full_training_dat_prior,'prior')[1]


               # #same for both post map and post all
               Dwin_post = get_production_probs(full_training_dat_post_all ,'post',cond='3')[0]
               feat_probs_post = get_production_probs(full_training_dat_post_all ,'post',cond='3')[1]

               print(Dwin_prior)
               print(feat_probs_prior)

               print(Dwin_post)
               print(feat_probs_post)

               print(main_data_formatted['rule_description'][i])

               # prior
               df_prior = mh_mcmc_sampler_res(productions, full_training_dat_prior,Dwin=Dwin,ground_truth=False, iter = 0)[0] # creating empty data for first initial rules
               prior_details = [] # details for prior rules are needed to run the map mcmc chain which starts with the map rule of the prior

               # post map chain
               df_post_map = mh_mcmc_sampler_res(productions, full_training_dat_post_map , Dwin=Dwin ,ground_truth=False, iter = 0)[0] # creating empty data for revised rules

               # post all chain
               df_post_all = mh_mcmc_sampler_res(productions, full_training_dat_post_all, Dwin=Dwin,iter = 0)[0] # creating empty data for all rules


               # mcmc chains
               # prior
               print('prior')
               for sample in range(n_1):
                    prior_df_and_details = mh_mcmc_sampler_res(productions,
                                                               full_training_dat_prior,
                                                               feat_probs=feat_probs_prior,
                                                               Dwin=Dwin_prior,
                                                               ground_truth=False,
                                                               iter = n_2,
                                                               type='prior',
                                                               out_penalizer=5)

                    df_prior = df_prior.append(prior_df_and_details[0])
                    prior_details.append(prior_df_and_details[1])


               #
                # post map
               # print(df_prior['rulestring'])
               max_a_posteriori_prior = df_prior['rulestring'].mode()[0]     # getting the mode of the prior chain first to initialise the post map chain
               map_index = np.where(df_prior["rulestring"] == max_a_posteriori_prior)[0][0]

               map_prior.append(max_a_posteriori_prior)


               # computing the accuracy of the other player
               # acc_partner = sum([a and b or not a and not b for a,b in zip(gt, eval(main_data_formatted['prior_partner'][i]))]) / 8
               print('post_map')
               for sample in range(n_1):
                    df_post_map = df_post_map.append(mh_mcmc_sampler_res(productions,
                                                             full_training_dat_post_map,
                                                             Dwin=Dwin_post,
                                                             feat_probs=feat_probs_post,
                                                             start=df_prior['rulestring'].mode()[0],
                                                             start_frame=prior_details[0][str(map_index)],
                                                             iter = n_2,
                                                             type='posterior_map',
                                                             prod_l=df_prior['prod_l'][map_index],
                                                             prod_d=df_prior['prod_d'][map_index],
                                                             out_penalizer=5)[0])
               #
               max_a_posteriori_post_map = df_post_map['rulestring'].mode()[0]
               map_post_map.append(max_a_posteriori_post_map)
               #
               #
               print('post_all')
               # post all (just based on all data points)
               for sample in range(n_1):
                    df_post_all = df_post_all.append(mh_mcmc_sampler_res(productions,
                                                           full_training_dat_post_all,
                                                           Dwin=Dwin_post,
                                                           feat_probs=feat_probs_post,
                                                           iter = n_2,
                                                           type='posterior_all',
                                                           out_penalizer=5)[0])

               #
               max_a_posteriori_post_all = df_post_all['rulestring'].mode()[0]
               map_post_all.append(max_a_posteriori_post_all)

               # calculating the probability that a scene is rule following for a specific trial and subject
               prob_gen_follow_rule_prior = []
               prob_gen_follow_rule_post_map = []
               prob_gen_follow_rule_post_all = []

               map_res_prior = []                # result checks whether this scene follows a rule (length of results equals n_1 * n_2)
               map_res_post_map = []
               map_res_post_all = []

               # evaluating the rules based on the generalization data shown to subjects
               for gen in generalizations:        # looping over all 8 generalization scenes


                    res_prior = []                # result checks whether this scene follows a rule (length of results equals n_1 * n_2)
                    res_post_map = []
                    res_post_all = []



                    # global X                # defining a global variable X
                    X = []                  # for each scene, X will include the objects (i.e., cones) of the scene

                    # looping over the number of objects (ie cones) in each scene
                    for i_3 in range(0, len(gen['ids'])):
                         # print(gen['contact'])
                         contact = check_structure(gen['contact'], i_3)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
                         # print(contact)
                         # getting the properties for each object (triangle / cone) in the scene
                         object = {"id": gen['ids'][i_3], "colour":  gen['colours'][i_3] , "size":  gen['sizes'][i_3], "xpos":  int(np.round(gen['xpos'][i_3])),
                         "ypos":  int(np.round(gen['ypos'][i_3])), "rotation": np.round(gen['rotations'][i_3],1), "orientation": compute_orientation(gen['rotations'][i_3])[0],
                         "grounded":  gen['grounded'][i_3], "contact":  contact}

                         X.append(object)   # appending the object to X which includes all objects for a scene
                    # print(X)
                    # print(ob.exists(lambda x1: ob.forall(lambda x2: ob.and_operator(ob.and_operator(ob.equal(x1,'red','colour'), ob.not_operator(ob.equal(x2, 'red', 'colour'))), ob.greater(x1,x2,'size')), X), X))
                    # evaluating all sampled rules against the scenes for each of the different mcmc chains and appending results
                    for rule in df_prior['rulestring']:
                         res_prior.append(eval(rule))
                         # print(eval(rule))
                    # print(X)
                    # print(ob.forall(lambda x1: ob.not_operator(ob.equal(x1, 'upright', 'orientation')), X))

                    for rule in df_post_map['rulestring']:
                         res_post_map.append(eval(rule))

                    for rule in df_post_all['rulestring']:
                         res_post_all.append(eval(rule))

                    map_res_prior.append(eval(max_a_posteriori_prior))
                    #
                    map_res_post_map.append(eval(max_a_posteriori_post_map))
                    map_res_post_all.append(eval(max_a_posteriori_post_all))

                    # computing the raw probabilities that the scenes follow a rule for each chain

                    p_follow_rule_prior = (1 / len(res_prior)) * sum(res_prior) # len(res) = number of rules; sum(res) = number of rules matching the scene
                    # print(p_follow_rule_prior)
                    prob_gen_follow_rule_prior.append(p_follow_rule_prior)

                    p_follow_rule_post_map = (1 / len(res_post_map)) * sum(res_post_map) # len(res) = number of rules; sum(res) = number of rules matching the scene
                    prob_gen_follow_rule_post_map.append(p_follow_rule_post_map)

                    p_follow_rule_post_all = (1 / len(res_post_all)) * sum(res_post_all) # len(res) = number of rules; sum(res) = number of rules matching the scene
                    prob_gen_follow_rule_post_all.append(p_follow_rule_post_all)

               map_prior_acc.append(compute_acc(gt, map_res_prior))
               map_post_map_acc.append(compute_acc(gt, map_res_post_map))
               map_post_all_acc.append(compute_acc(gt, map_res_post_all))


               # return 1
               # fitting tau to the data using a generic minimize function from scipy.optimize for all chains

               ####################### SINGLE TRIALS ##########################################
               # prior chain
               raw_probs_prior.append(prob_gen_follow_rule_prior)  # only used for single trial examples
               raw_probs_all_trials_prior.append(prob_gen_follow_rule_prior)
               prior_resp_all_trials_prior.append(prior_response)
               fitted_tau_prior = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(prob_gen_follow_rule_prior,prior_response), method='L-BFGS-B')
               fitted_taus_prior.append(fitted_tau_prior.x[0])
               fitted_results_mod_prior = fitted_probs(fitted_tau_prior.x[0], prob_gen_follow_rule_prior, prior_response)
               select_probs_prior.append(fitted_results_mod_prior[0])
               ll_model_prior.append(fitted_results_mod_prior[1])
               BICs_model_prior.append(-2 * fitted_results_mod_prior[1] + 1 * np.log(8))



               # repeating the above for post map chain
               raw_probs_post_map.append(prob_gen_follow_rule_post_map)  # only used for single trial examples
               raw_probs_all_trials_post_map.append(prob_gen_follow_rule_post_map)
               post_resp_all_trials_post_map.append(posterior_response)
               fitted_tau_post_map = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(prob_gen_follow_rule_post_map,posterior_response), method='L-BFGS-B')
               fitted_taus_post_map.append(fitted_tau_post_map.x[0])
               fitted_results_mod_post_map = fitted_probs(fitted_tau_post_map.x[0], prob_gen_follow_rule_post_map, posterior_response)
               select_probs_post_map.append(fitted_results_mod_post_map[0])
               ll_model_post_map.append(fitted_results_mod_post_map[1])
               BICs_model_post_map.append(-2 * fitted_results_mod_post_map[1] + 1 * np.log(8))


               # # and post all chain
               raw_probs_post_all.append(prob_gen_follow_rule_post_all)
               raw_probs_all_trials_post_all.append(prob_gen_follow_rule_post_all)
               post_resp_all_trials_post_all.append(posterior_response)

               fitted_tau_post_all = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(prob_gen_follow_rule_post_all,posterior_response), method='L-BFGS-B')
               fitted_taus_post_all.append(fitted_tau_post_all.x[0])
               fitted_results_mod_post_all = fitted_probs(fitted_tau_post_all.x[0], prob_gen_follow_rule_post_all, posterior_response)
               select_probs_post_all.append(fitted_results_mod_post_all[0])
               ll_model_post_all.append(fitted_results_mod_post_all[1])
               BICs_model_post_all.append(-2 * fitted_results_mod_post_all[1] + 1 * np.log(8))

               # # computing results of baseline model prior for a single trial
               baseline_probs = [.5, .5, .5, .5, .5, .5, .5, .5]
               fitted_results_baseline = fitted_probs(1, baseline_probs, prior_response)
               ll_baseline_prior.append(fitted_results_baseline[1])
               BICs_baseline_prior.append(-2 * fitted_results_baseline[1] + 0 * np.log(8))

               fitted_results_baselin_post_map = fitted_probs(1, baseline_probs, posterior_response)
               ll_baseline_post_map.append(fitted_results_baselin_post_map[1])
               BICs_baseline_post_map.append(-2 * fitted_results_baselin_post_map[1] + 0 * np.log(8))

               fitted_results_baselin_post_all = fitted_probs(1, baseline_probs, posterior_response)
               ll_baseline_post_all.append(fitted_results_baselin_post_all[1])   # same for basline
               BICs_baseline_post_all.append(-2 * fitted_results_baselin_post_all[1] + 0 * np.log(8))

               ####################### ALL TRIALS FOR ONE SUBJECT ##########################################
               if len(fitted_taus_prior) % n_trials == 0:
                    # print(n_trials)
                    # print('hi')
                    # print(n_trials)
                    raw_probs_all_trials_1 = [prob for sublist in raw_probs_all_trials_prior[i-n_trials+1:] for prob in sublist]
                    prior_resp_all_trials_1 = [response for sublist in prior_resp_all_trials_prior[i-n_trials+1:] for response in sublist]


                    overall_fitted_tau = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_trials_1,prior_resp_all_trials_1), method='L-BFGS-B')
                    fitted_results_all_trials_mod = fitted_probs(overall_fitted_tau.x[0], raw_probs_all_trials_1, prior_resp_all_trials_1)

                    baseline_probs_all_trials = [.5] * 8 * n_trials
                    fitted_results_all_trials_baseline = fitted_probs(1, baseline_probs_all_trials, prior_resp_all_trials_1)

                    raw_probs_all_trials_post_map_1 = [prob for sublist in raw_probs_all_trials_post_map[i-n_trials+1:] for prob in sublist]
                    post_resp_all_trials_post_map_1 = [response for sublist in post_resp_all_trials_post_map[i-n_trials+1:] for response in sublist]

                    overall_fitted_tau_post_map = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_trials_post_map_1, post_resp_all_trials_post_map_1), method='L-BFGS-B')
                    fitted_results_all_trials_mod_post_map = fitted_probs(overall_fitted_tau_post_map.x[0], raw_probs_all_trials_post_map_1 ,  post_resp_all_trials_post_map_1)

                    raw_probs_all_trials_post_all_1 = [prob for sublist in raw_probs_all_trials_post_all[i-n_trials+1:] for prob in sublist]
                    post_resp_all_trials_post_all_1 = [response for sublist in post_resp_all_trials_post_all[i-n_trials+1:] for response in sublist]

                    overall_fitted_tau_post_all = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_trials_post_all_1, post_resp_all_trials_post_all_1), method='L-BFGS-B')
                    fitted_results_all_trials_mod_post_all = fitted_probs(overall_fitted_tau_post_all.x[0], raw_probs_all_trials_post_all_1 ,  post_resp_all_trials_post_all_1)


                    baseline_probs_all_trials = [.5] * 8 * n_trials
                    fitted_results_all_trials_baseline_post_map = fitted_probs(1, baseline_probs_all_trials, post_resp_all_trials_post_map_1)
                    fitted_results_all_trials_baseline_post_all = fitted_probs(1, baseline_probs_all_trials, post_resp_all_trials_post_all_1)

                    #
                    prior_label = compare_BIC(fitted_results_all_trials_mod[1], fitted_results_all_trials_baseline[1], 8*n_trials)
                    post_map_label = compare_BIC(fitted_results_all_trials_mod_post_map[1], fitted_results_all_trials_baseline_post_map[1], 8*n_trials)
                    post_all_label = compare_BIC(fitted_results_all_trials_mod_post_all[1], fitted_results_all_trials_baseline_post_all[1], 8*n_trials)


                    if len(fitted_taus_prior) % len(main_data_formatted['rule_name']) == 0:

                         # prior
                         # print('potato')
                         raw_probs_all_subjects = [prob for sublist in raw_probs_all_trials_prior for prob in sublist]
                         prior_resp_all_subjects = [response for sublist in prior_resp_all_trials_prior for response in sublist]

                         fitted_tau_all_subjects_prior = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_subjects,prior_resp_all_subjects), method='L-BFGS-B')
                         fitted_results_all_subjects_prior = fitted_probs(fitted_tau_all_subjects_prior.x[0], raw_probs_all_subjects,prior_resp_all_subjects)


                         fitted_taus_all_subjects_prior = []
                         fitted_taus_all_subjects_prior.append(fitted_tau_all_subjects_prior.x[0])

                         select_probs_all_subjects_prior = []
                         select_probs_all_subjects_prior.append(fitted_results_all_subjects_prior[0])

                         ll_model_all_subjects_prior = []
                         ll_model_all_subjects_prior.append(fitted_results_all_subjects_prior[1])

                         BICs_model_all_subjects_prior = []
                         BICs_model_all_subjects_prior.append(-2 * fitted_results_all_subjects_prior[1] + 1 * np.log(8 * len(main_data_formatted['rule_name'])))


                         # post map

                         raw_probs_all_subjects_post_map = [prob for sublist in raw_probs_all_trials_post_map for prob in sublist]
                         resp_all_subjects_post_map = [response for sublist in post_resp_all_trials_post_map for response in sublist]

                         fitted_tau_all_subjects_post_map = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_subjects_post_map,resp_all_subjects_post_map), method='L-BFGS-B')
                         fitted_results_all_subjects_post_map = fitted_probs(fitted_tau_all_subjects_post_map.x[0], raw_probs_all_subjects_post_map,resp_all_subjects_post_map)

                         fitted_taus_all_subjects_post_map = []
                         fitted_taus_all_subjects_post_map.append(fitted_tau_all_subjects_post_map.x[0])

                         select_probs_all_subjects_post_map = []
                         select_probs_all_subjects_post_map.append(fitted_results_all_subjects_post_map[0])

                         ll_model_all_subjects_post_map = []
                         ll_model_all_subjects_post_map.append(fitted_results_all_subjects_post_map[1])

                         BICs_model_all_subjects_post_map = []
                         BICs_model_all_subjects_post_map.append(-2 * fitted_results_all_subjects_post_map[1] + 1 * np.log(8 * len(main_data_formatted['rule_name'])))

                         # post all

                         raw_probs_all_subjects_post_all = [prob for sublist in raw_probs_all_trials_post_all for prob in sublist]
                         resp_all_subjects_post_all = [response for sublist in post_resp_all_trials_post_all for response in sublist]

                         fitted_tau_all_subjects_post_all = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_subjects_post_all,resp_all_subjects_post_all), method='L-BFGS-B')
                         fitted_results_all_subjects_post_all = fitted_probs(fitted_tau_all_subjects_post_all.x[0], raw_probs_all_subjects_post_all,resp_all_subjects_post_all)

                         fitted_taus_all_subjects_post_all = []
                         fitted_taus_all_subjects_post_all.append(fitted_tau_all_subjects_post_all.x[0])

                         select_probs_all_subjects_post_all = []
                         select_probs_all_subjects_post_all.append(fitted_results_all_subjects_post_all[0])

                         ll_model_all_subjects_post_all = []
                         ll_model_all_subjects_post_all.append(fitted_results_all_subjects_post_all[1])

                         BICs_model_all_subjects_post_all = []
                         BICs_model_all_subjects_post_all.append(-2 * fitted_results_all_subjects_post_all[1] + 1 * np.log(8 * len(main_data_formatted['rule_name'])))


                         # baseline

                         baseline_probs_all_subjects = [.5] * 8 * len(main_data_formatted['rule_name'])
                         fitted_results_all_subjects_baseline = fitted_probs(1, baseline_probs_all_subjects, prior_resp_all_subjects)


                         ll_baseline_all_subjects_prior = []
                         #ll_baseline_all_subjects_prior.append(fitted_results_all_subjects_baseline[1])

                         ll_baseline_all_subjects_prior.append(fitted_results_all_subjects_baseline[1])

                         BICs_baseline_all_subjects_prior = []
                         BICs_baseline_all_subjects_prior.append(-2 * fitted_results_all_subjects_baseline[1])
                         # print(BICs_baseline_all_subjects_prior)

                         # BICs_baseline_all_subjects_prior = []
                         # BICs_baseline_all_subjects_prior.append(-2 * fitted_results_all_subjects_baseline[1] + 0 * np.log(40 * len(fitted_taus_prior)))

                         # overall accuracy
                         gt_all_subjects = gt * len(fitted_taus_prior) # ground truth

                         prior_mod_select_all_subj = hard_max_selections(raw_probs_all_subjects)
                         post_map_mod_select_all_subj = hard_max_selections(raw_probs_all_subjects_post_map)
                         post_all_mod_select_all_subj = hard_max_selections(raw_probs_all_subjects_post_all)


                         prior_acc_all_subj = sum([a and b or not a and not b for a, b in zip(gt_all_subjects, prior_mod_select_all_subj)]) / len(prior_mod_select_all_subj)
                         post_map_acc_all_subj = sum([a and b or not a and not b for a, b in zip(gt_all_subjects, post_map_mod_select_all_subj)]) / len(post_map_mod_select_all_subj)
                         post_all_acc_all_subj = sum([a and b or not a and not b for a, b in zip(gt_all_subjects, post_all_mod_select_all_subj)]) / len(post_all_mod_select_all_subj)


                         # print(len(fitted_taus_prior))








                    for trial in range(n_trials):

                         fitted_taus_all_trials_prior.append(overall_fitted_tau.x[0])
                         select_probs_all_trials_prior.append(fitted_results_all_trials_mod[0])
                         ll_model_all_trials_prior.append(fitted_results_all_trials_mod[1])
                         BICs_model_all_trials_prior.append(-2 * fitted_results_all_trials_mod[1] + 1 * np.log(8*n_trials))
                         ll_baseline_all_trials_prior.append(fitted_results_all_trials_baseline[1])
                         BICs_baseline_all_trials_prior.append(-2 * fitted_results_all_trials_baseline[1] + 0 * np.log(8*n_trials))
                         raw_probs_all_trials_one_list_prior.append(raw_probs_all_trials_prior)

                         #
                         fitted_taus_all_trials_post_map.append(overall_fitted_tau_post_map.x[0])
                         select_probs_all_trials_post_map.append(fitted_results_all_trials_mod_post_map[0])
                         ll_model_all_trials_post_map.append(fitted_results_all_trials_mod_post_map[1])
                         BICs_model_all_trials_post_map.append(-2 * fitted_results_all_trials_mod_post_map[1] + 1 * np.log(8*n_trials))
                         ll_baseline_all_trials_post_map.append(fitted_results_all_trials_baseline_post_map[1])
                         BICs_baseline_all_trials_post_map.append(-2 * fitted_results_all_trials_baseline_post_map[1] + 0 * np.log(8*n_trials))
                         raw_probs_all_trials_one_list_post_map.append(raw_probs_all_trials_post_map)



                         fitted_taus_all_trials_post_all.append(overall_fitted_tau_post_all.x[0])
                         select_probs_all_trials_post_all.append(fitted_results_all_trials_mod_post_all[0])
                         ll_model_all_trials_post_all.append(fitted_results_all_trials_mod_post_all[1])
                         BICs_model_all_trials_post_all.append(-2 * fitted_results_all_trials_mod_post_all[1] + 1 * np.log(8*n_trials))
                         ll_baseline_all_trials_post_all.append(fitted_results_all_trials_baseline_post_map[1])
                         BICs_baseline_all_trials_post_all.append(-2 * fitted_results_all_trials_baseline_post_map[1] + 0 * np.log(8*n_trials))
                         raw_probs_all_trials_one_list_post_all.append(raw_probs_all_trials_post_all)

                         prior_labels.append(prior_label)
                         post_map_labels.append(post_map_label)
                         post_all_labels.append(post_all_label)






                    # computing accuracy of model predictions using hard maximization for selection probs
                    gt_all = gt * n_trials # ground truth

                    prior_mod_select = hard_max_selections(raw_probs_all_trials_1)
                    post_map_mod_select = hard_max_selections(raw_probs_all_trials_post_map_1)
                    post_all_mod_select = hard_max_selections(raw_probs_all_trials_post_all_1)

                    prior_acc = sum([a and b or not a and not b for a, b in zip(gt_all, prior_mod_select)]) / len(prior_mod_select)
                    post_map_acc = sum([a and b or not a and not b for a, b in zip(gt_all, post_map_mod_select)]) / len(post_map_mod_select)
                    post_all_acc = sum([a and b or not a and not b for a, b in zip(gt_all, post_all_mod_select)]) / len(post_all_mod_select)

                    low_bound = 0
                    up_bound = 8
                    for acc in range(n_trials):
                         prior_accs.append(prior_acc)
                         post_map_accs.append(post_map_acc)
                         post_all_accs.append(post_all_acc)
                         # single accuracies for prior
                         prior_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, prior_mod_select[low_bound:up_bound])]) / 8)
                         post_map_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, post_map_mod_select[low_bound:up_bound])]) / 8)
                         post_all_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, post_all_mod_select[low_bound:up_bound])]) / 8)
                         low_bound+=8
                         up_bound+=8






               # # creating a unique csv for each subject for each trial to store the outcome of the mcmc chains
               # df_prior.to_csv("sampled_rules_10_05/"+token_id + "_" + rule_name + "_prior.csv")       # storing a separate df including rules for each participant
               # df_post_map.to_csv("sampled_rules_10_05/"+token_id + "_" + rule_name + "_post_map.csv")
               # df_post_all.to_csv("sampled_rules_10_05/"+token_id + "_" + rule_name + "_post_all.csv")
               # print(raw_probs_all_trials_one_list_prior)
               i+=1 # proceeding to next trial
          #
          # ax = sns.countplot(x="rulestring", data=df_prior)
          # plt.figure(figsize=(2, 2))
          # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize = 8)
          # plt.tight_layout()
          # plt.show()

          # computing overall tau and BIC
          #  baseline_probs_all = [.5, .5, .5, .5, .5, .5, .5, .5] * 20
          #  fitted_results_baseline_all = fitted_probs(1, baseline_probs_all, [response for sublist in prior_resp_all_trials_prior for response in sublist])
          #  ll_baseline_prior_all = (fitted_results_baseline_all[1])
          #  BICs_baseline_all = -2 * fitted_results_baseline_all[1] + 0 * np.log(8*20)

          ################ APPENDING ALL DATA TO MAIN DATA FRAME ########################


          # print(raw_probs_all_trials_one_list_prior)
          # prior trial specific data
          main_data_formatted = main_data_formatted
          main_data_formatted['raw_probs_prior'] = raw_probs_prior
          main_data_formatted['fitted_tau_prior'] = fitted_taus_prior
          main_data_formatted['log_ll_model_prior'] = ll_model_prior
          main_data_formatted['log_ll_baseline_prior'] = ll_baseline_prior
          main_data_formatted['BIC_model_prior'] = BICs_model_prior
          main_data_formatted['BIC_baseline_prior'] = BICs_baseline_prior
          main_data_formatted['select_probs_fitted_tau_prior'] = select_probs_prior

          # prior data across trials
          main_data_formatted['raw_probs_all_trials_prior'] = raw_probs_all_trials_one_list_prior
          main_data_formatted['fitted_tau_all_trials_prior'] = fitted_taus_all_trials_prior
          main_data_formatted['log_ll_model_all_trials_prior'] = ll_model_all_trials_prior
          main_data_formatted['log_ll_baseline_all_trials_prior'] = ll_baseline_all_trials_prior
          main_data_formatted['BIC_model_all_trials_prior'] = BICs_model_all_trials_prior
          main_data_formatted['BIC_baseline_all_trials_prior'] = BICs_baseline_all_trials_prior
          main_data_formatted['select_probs_fitted_tau_all_trials_prior'] = select_probs_all_trials_prior

          print('giantpotato')
          # print(fitted_taus_all_subjects_prior)
          # print(len(select_probs_all_subjects_prior))
          # print(select_probs_all_subjects_prior)
          # print(len(select_probs_all_subjects_prior[0]))
          # print(ll_model_all_subjects_prior)
          # print(BICs_model_all_subjects_prior)
          # print(ll_baseline_all_subjects_prior)
          # print(BICs_baseline_all_subjects_prior)
          main_data_formatted['fitted_taus_all_subjects_prior'] = fitted_taus_all_subjects_prior * 450
          # main_data_formatted['select_probs_all_subjects_prior'] = [[select_probs_all_subjects_prior]] * 450
          # print(len(select_probs_all_subjects_prior * 450))
          main_data_formatted['ll_model_all_subjects_prior'] = ll_model_all_subjects_prior* 450
          main_data_formatted['BICs_model_all_subjects_prior'] = BICs_model_all_subjects_prior* 450
          main_data_formatted['ll_baseline_all_subjects_prior'] = ll_baseline_all_subjects_prior* 450
          main_data_formatted['BICs_baseline_all_subjects_prior'] = BICs_baseline_all_subjects_prior* 450

          # post map trial specific data
          main_data_formatted['raw_probs_post_map'] = raw_probs_post_map
          main_data_formatted['fitted_tau_post_map'] = fitted_taus_post_map
          main_data_formatted['log_ll_model_post_map'] = ll_model_post_map
          main_data_formatted['log_ll_baseline_post_map'] = ll_baseline_post_map
          main_data_formatted['BIC_model_post_map'] = BICs_model_post_map
          main_data_formatted['BIC_baseline_post_map'] = BICs_baseline_post_map
          main_data_formatted['select_probs_fitted_tau_post_map'] = select_probs_post_map


          # post map data across trials
          main_data_formatted['raw_probs_all_trials_post_map'] = raw_probs_all_trials_one_list_post_map
          main_data_formatted['fitted_tau_all_trials_post_map'] = fitted_taus_all_trials_post_map
          main_data_formatted['log_ll_model_all_trials_post_map'] = ll_model_all_trials_post_map
          main_data_formatted['log_ll_baseline_all_trials_post_map'] = ll_baseline_all_trials_post_map
          main_data_formatted['BIC_model_all_trials_post_map'] = BICs_model_all_trials_post_map
          main_data_formatted['BIC_baseline_all_trials_post_map'] = BICs_baseline_all_trials_post_map
          main_data_formatted['select_probs_fitted_tau_all_trials_post_map'] = select_probs_post_map
          #
          # print(fitted_taus_all_subjects_post_map)
          # print(len(select_probs_all_subjects_post_map))
          # print(ll_model_all_subjects_post_map)
          # print(BICs_model_all_subjects_post_map)
          # print(ll_baseline_all_subjects_prior)
          # print(BICs_baseline_all_subjects_prior)
          main_data_formatted['fitted_taus_all_subjects_post_map'] = fitted_taus_all_subjects_post_map *   450
          # main_data_formatted['select_probs_all_subjects_post_map'] = select_probs_all_subjects_post_map
          main_data_formatted['ll_model_all_subjects_post_map'] = ll_model_all_subjects_post_map * 450
          main_data_formatted['BICs_model_all_subjects_post_map'] = BICs_model_all_subjects_post_map* 450
          main_data_formatted['ll_baseline_all_subjects_post_map'] = ll_baseline_all_subjects_prior * 450
          main_data_formatted['BICs_baseline_all_subjects_post_map'] = BICs_baseline_all_subjects_prior* 450
          #
          # #
          # # post all specific data
            # post map trial specific data
          main_data_formatted['raw_probs_post_all'] = raw_probs_post_all
          main_data_formatted['fitted_tau_post_all'] = fitted_taus_post_all
          main_data_formatted['log_ll_model_post_all'] = ll_model_post_all
          main_data_formatted['log_ll_baseline_post_all'] = ll_baseline_post_all
          main_data_formatted['BIC_model_post_all'] = BICs_model_post_all
          main_data_formatted['BIC_baseline_post_all'] = BICs_baseline_post_all
          main_data_formatted['select_probs_fitted_tau_post_all'] = select_probs_post_all


          # post map data across trials
          main_data_formatted['raw_probs_all_trials_post_all'] = raw_probs_all_trials_one_list_post_all
          main_data_formatted['fitted_tau_all_trials_post_all'] = fitted_taus_all_trials_post_all
          main_data_formatted['log_ll_model_all_trials_post_all'] = ll_model_all_trials_post_all
          main_data_formatted['log_ll_baseline_all_trials_post_all'] = ll_baseline_all_trials_post_all
          main_data_formatted['BIC_model_all_trials_post_all'] = BICs_model_all_trials_post_all
          main_data_formatted['BIC_baseline_all_trials_post_all'] = BICs_baseline_all_trials_post_all
          main_data_formatted['select_probs_fitted_tau_all_trials_post_all'] = select_probs_all_trials_post_all
          # print(fitted_taus_all_subjects_post_all)
          # print(select_probs_all_subjects_post_all)
          # print(ll_model_all_subjects_post_all)
          # print(BICs_model_all_subjects_post_all)
          # print(ll_baseline_all_subjects_prior)
          # print(BICs_baseline_all_subjects_prior)
          main_data_formatted['fitted_taus_all_subjects_post_all'] = fitted_taus_all_subjects_post_all * 450
          # main_data_formatted['select_probs_all_subjects_post_all'] = select_probs_all_subjects_post_all
          main_data_formatted['ll_model_all_subjects_post_all'] = ll_model_all_subjects_post_all * 450
          main_data_formatted['BICs_model_all_subjects_post_all'] = BICs_model_all_subjects_post_all* 450
          main_data_formatted['ll_baseline_all_subjects_post_all'] = ll_baseline_all_subjects_prior* 450
          main_data_formatted['BICs_baseline_all_subjects_post_all'] = BICs_baseline_all_subjects_prior * 450

          # labels comparing baseline performance with mcmc performance (1 = mcmc wins for this subject)
          main_data_formatted['prior_label'] = prior_labels
          main_data_formatted['post_map_label'] = post_map_labels
          main_data_formatted['post_all_label'] = post_all_labels

          # accuracy of mcmc model predictions
          main_data_formatted['prior_acc'] = prior_accs
          main_data_formatted['prior_acc_single'] = prior_accs_single
          main_data_formatted['post_map_acc'] = post_map_accs
          main_data_formatted['post_map_acc_single'] = post_map_accs_single
          main_data_formatted['post_all_acc'] = post_all_accs
          main_data_formatted['post_all_acc_single'] = post_all_accs_single




          # maps
          main_data_formatted['map_prior'] = map_prior
          main_data_formatted['map_post_map'] = map_post_map
          main_data_formatted['map_post_all'] = map_post_all

          # map accs
          main_data_formatted['map_prior_acc'] = map_prior_acc
          main_data_formatted['map_post_map_acc'] = map_post_map_acc
          main_data_formatted['map_post_all_acc'] = map_post_all_acc


          # all subjects average accuracy single tau
          main_data_formatted['all_sub_prior_acc'] = prior_acc_all_subj
          main_data_formatted['all_sub_post_map_acc'] = post_map_acc_all_subj
          main_data_formatted['all_sub_post_all_acc'] = post_all_acc_all_subj



          main_data_formatted.to_csv('boolean_adaptive_1000_survive/exp_3/boolean_adaptive_pcfg_5_rules_5_penalisation_cond_3' + str(rep) + '_round_7.csv')   # writing main data to new csv file including all relevant data for analysis

          rep+=1

predicted_selections(main_data_formatted, rules_dict, trial_counts, n_rep=25)






# # # sampling rules absed on non-deterministic likelihood (similar to Goodman et al., 2008)
#
# for i in range(1):
#      df_rules_1 = df_rules_1.append(mh_mcmc_sampler(productions, data, ground_truth=False, iter = 100)[0])
# #
 # # #
# #
# print(cat)

# print(df_rules_1["rulestring"])
# print(df_rules_1["li"])
# df_rules_1.to_csv('rules_subj_test.csv')
# #
#
# #
# # # sampling rules absed on determinsitic likelihood (identical to procedure in Bramley et al. (2018)
# df_rules_2 = sample_rules(productions, data, ground_truth, iter=2)
# ax_2 = sns.countplot(x="rulestring", data=df_rules_2)
# plt.show()
# print(df_rules_2)
# print(df_rules_2["rulestring"][0])
#

          # ax = sns.countplot(x="rulestring", data=df)

          #
          # plt.figure(figsize=(2, 2))
          # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize = 8)
          # plt.tight_layout()
          # plt.show()
