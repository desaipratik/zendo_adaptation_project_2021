"""
Created on Sun Mar 15 15:01:21 2020

@author: jan-philippfranken
"""
###############################################################################
########################### Main File #########################################
###############################################################################


####################### General imports #######################################
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from statistics import mean
from scipy.optimize import minimize
print(pd.__version__)

####################### Custom imports ########################################
from pcfg_generate_rules_subtree_regeneration_restricted_grammar import mh_mcmc_sampler_res
from pcfg_generate_rules_subtree_regeneration import mh_mcmc_sampler
from pruned_pcfg import pcfg_generator
from transform_functions import compute_orientation, check_structure, check_change_mult_feat, check_feat_change, check_change_impact_single_feat, check_feat_ident, transform_xpos, transform_ypos
from tau_ml_estimation import get_tau, fitted_probs, compare_BIC, hard_max_selections, compute_acc, compute_distance



###################### Preliminaries ##########################################
ob = pcfg_generator()                                         # instantiating pcfg generator
random.seed(1)                              # setting random.seed to allow replication of mcmc chain
main_data_formatted = pd.read_csv('main_data_formatted.csv')  # getting the preprocessed data file


####################### Grammar ##############################################
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

# summarizing grammar in dictionary
productions = {"S": S, "A": A, "B": B, "C": C, "D": D, "E": E, "G": G, "H": H, "I": I, "J": J, "K": K, "L": L, "M": M}

####################### Sampling Algorithm #########################################
def predicted_selections(main_data_formatted, n_1=1, n_2=100):  # computes the ll for getting participants responses to initial generalizations (n_1 * n_2 determines number of iterations)
     rep = 0

     for repeat in np.arange(10):
          i = 0                          # index over trials
          gt = [1,1,1,1,0,0,0,0]

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
          select_probs_post_map = []
          raw_probs_all_trials_post_map = []
          raw_probs_all_trials_one_list_post_map = []
          select_probs_all_trials_post_map = []
          post_resp_all_trials_post_map = []
          fitted_taus_all_trials_post_map = []
          ll_model_all_trials_post_map = []
          BICs_model_all_trials_post_map = []
          ll_baseline_all_trials_post_map = []
          BICs_baseline_all_trials_post_map = []

          # repeating the above for subjects posteriores based on all 16 data points
          select_probs_post_all = []
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
          post_all_accs = []

          # labels (= 1 if model is better fit for subject and 0 if baseline is better fit for subject)
          prior_labels = []
          post_map_labels = []
          post_all_labels = []

          # maps
          map_prior = []
          map_post_map = []
          map_post_all = []

          # maps
          map_prior_acc = []
          map_post_map_acc = []
          map_post_all_acc = []

          #  looping over each trial for each subject (n_trials * n_subjects iterations and run mcmc chains)
          for data in main_data_formatted['data'][:100]:


               # getting name and id to create a unique csv file for each mcmc chain for each subject and trial
               rule_name = main_data_formatted['rule_name'][i]
               token_id = main_data_formatted['token_id'][i]

               # getting subjects prior and posterior responses
               prior_response = eval(main_data_formatted['prior'][i])
               posterior_response = eval(main_data_formatted['posterior'][i])

               # getting the data
               trials = eval(data)[:8]           # trials = scenes created by participants
               generalizations = eval(data)[8:]  # generalizations = predefined scenes that participants need to select
               full_data = eval(data)            # full data used for the third mcmc chain combining trials and generalizations
               #
               #
               grounded = []
               colours = []
               sizes = []
               orientations = []
               follow = []
               contact = []
               xpos = []
               ypos = []
               rotations = []

               for trial in trials:

                    colours.append(trial['colours'])
                    sizes.append(trial['sizes'])
                    grounded.append(trial['grounded'])
                    follow.append(trial['follow_rule'])
                    # contact.append(trial['contact'])
                    # xpos.append(trial['xpos'])
                    # ypos.append(trial['ypos'])
                    # rotations.append(trial['rotations'])
                    orientations.append(trial['orientations'])

               #
               # restricted to one bound variable including only non-continuous features
               # kkk = check_change_mult_feat([check_feat_change(colours),
               #                         check_feat_change(sizes),
               #                         check_feat_change(orientations),
               #                         check_feat_change(grounded)],
               #                         follow)

                 #including all
               # kkk = check_change_mult_feat([check_feat_change(colours),
               #                         check_feat_change(sizes),
               #                         check_feat_change(orientations),
               #                         check_feat_change(grounded),
               #                         check_feat_change(transform_xpos(xpos)),
               #                        check_feat_change(transform_ypos(ypos)),
               #                        check_feat_change(orientations),
               #                            check_feat_change(contact,'sets')],
               #                         follow)


              #  s = [sum(check_change_impact_single_feat(follow,check_feat_change(colours)))+sum(list(kkk['0'].values())),
              # sum(check_change_impact_single_feat(follow,check_feat_change(sizes)))+sum(list(kkk['1'].values())),
              # sum(check_change_impact_single_feat(follow,check_feat_change(orientations)))+sum(list(kkk['2'].values())),
              # sum(check_change_impact_single_feat(follow,check_feat_change(grounded)))+sum(list(kkk['3'].values()))]

              #  feat_scores = [sum(check_change_impact_single_feat(follow,check_feat_change(colours,follow)['feat_change'])),
              # sum(check_change_impact_single_feat(follow,check_feat_change(sizes,follow)['feat_change'])),
              # sum(check_change_impact_single_feat(follow,check_feat_change(orientations,follow)['feat_change'])),
              # sum(check_change_impact_single_feat(follow,check_feat_change(grounded,follow)['feat_change']))]

              # st =  [sum(list(kkk['0'].values())),sum(list(kkk['1'].values())),sum(list(kkk['2'].values())),
              #       sum(list(kkk['3'].values())),sum(list(kkk['4'].values())),sum(list(kkk['5'].values())),
              #       sum(list(kkk['6'].values())),sum(list(kkk['7'].values()))]

               feat_scores = [check_feat_change(colours,follow)['sum_values'],
                             check_feat_change(sizes,follow)['sum_values'],
                             check_feat_change(orientations,follow)['sum_values'],
                             check_feat_change(grounded,follow)['sum_values']]

               val_scores = [check_feat_change(colours,follow)['val_change'],
                             check_feat_change(sizes,follow)['val_change'],
                             check_feat_change(orientations,follow)['val_change'],
                             check_feat_change(grounded,follow)['val_change']]


               val_defaults = [{'red':1,'blue':1,'green':1},
                               {'1':1,'2':1,'3':1},
                               {'upright':1, 'lhs':1, 'rhs':1, 'strange':1},
                               {'no':1,'yes':1}]

               for val_score in val_scores:
                   # print(val_score)
                   for key in val_score.keys():
                        # print(key)
                        val_defaults[val_scores.index(val_score)][str(key)] += val_score[key]


               # print(val_defaults)

               color_probs = list(val_defaults[0].values())
               color_probs = [float(zzz)/sum(color_probs) for zzz in color_probs]
               size_probs = list(val_defaults[1].values())
               size_probs = [float(zzz)/sum(size_probs) for zzz in size_probs]
               orientation_probs = list(val_defaults[2].values())
               orientation_probs = [float(zzz)/sum(orientation_probs) for zzz in orientation_probs]
               grounded_probs = list(val_defaults[3].values())
               grounded_probs = [float(zzz)/sum(grounded_probs) for zzz in grounded_probs]

               feat_probs = [color_probs,size_probs,[0],[0],[0],orientation_probs,grounded_probs]

     #         print(kkk)
     #           k = []
     #           if min(feat_scores) < 0:
     #               for zz in feat_scores:
     #                   zz+= -(min(feat_scores))
     #                   if zz == 0:
     #                        zz = 1
     #                   k.append(zz)
     #           else:
     #               k = feat_scores
     #           # print(s)
     # #         print(k)
     # #           print(k)
     #           if sum(k) == 0:
     #               k=[.25]*4
     #           k = [float(zzz)/sum(k) for zzz in k]

               # k= [0.2711430171093845, 0.147129249234712, 0.104002196071551, 0.0955371952030816, 0.104421543379247, 0.154718273384734, 0.12304852561729]
               print(main_data_formatted['rule_description'][i])

               # print(val_defaults)
               # print(k)

               Dwin = [1+feat_scores[0],1+feat_scores[1],0,0,0,1+feat_scores[2],1+feat_scores[3]]
               # print(Dwin)
               # print(Dwin)
               Dwin = [float(zzz)/sum(Dwin) for zzz in Dwin]

               print(Dwin)
               print(feat_probs)

               # print(Dwin)
               # print(feat_probs)
               # print(Dwin)
               # print(s)

               # Cwin = [0.233852890069965, 0.184656152815279, 0.182015761909768, 0.208453641617845, 0.191021553587143]
               # Cwin = [float(zzz)/sum(Cwin) for zzz in Cwin]
               # Cwin2=Cwin+[0]
               # print(Cwin)

               # k = k[:7]
               # k = [float(zzz)/sum(k) for zzz in k]


               # print(Cwin)
               # print(k)
               # k= np.repeat(1/len(productions["D"]), len(productions["D"]))

               # feat_probs=[[0.333333333,.333333333,.333333333],[0.333333333,.333333333,.333333333],[0],[0],[0],[.25,.25,.25,.25],[.5,.5]]
               # print(feat_probs)
               # creating data frames, one for each different mcmc chain
               # prior chain
               df_prior = mh_mcmc_sampler_res(productions, trials,Dwin=Dwin,feat_probs=feat_probs,ground_truth=False, iter = 0)[0] # creating empty data for first initial rules
               prior_details = [] # details for prior rules are needed to run the map mcmc chain which starts with the map rule of the prior

               # post map chain
               df_post_map = mh_mcmc_sampler_res(productions, trials,Dwin=Dwin,feat_probs=feat_probs ,ground_truth=False, iter = 0)[0] # creating empty data for revised rules

               # post all chain
               df_post_all = mh_mcmc_sampler_res(productions, trials, Dwin=Dwin,feat_probs=feat_probs ,iter = 0)[0] # creating empty data for all rules


               # mcmc chains
               # prior
               for sample in range(n_1):
                    prior_df_and_details = mh_mcmc_sampler_res(productions,
                                                               trials,
                                                               Dwin=Dwin,feat_probs=feat_probs,
                                                               ground_truth=False,
                                                               iter = n_2,
                                                               type='prior',
                                                               out_penalizer=5)

                    df_prior = df_prior.append(prior_df_and_details[0])
                    prior_details.append(prior_df_and_details[1])


               #
                # post map
               max_a_posteriori_prior = df_prior['rulestring'].mode()[0]     # getting the mode of the prior chain first to initialise the post map chain
               map_index = np.where(df_prior["rulestring"] == max_a_posteriori_prior)[0][0]

               map_prior.append(max_a_posteriori_prior)


               # computing the accuracy of the other player
               acc_partner = sum([a and b or not a and not b for a,b in zip(gt, eval(main_data_formatted['prior_partner'][i]))]) / 8

               for sample in range(n_1):
                    df_post_map = df_post_map.append(mh_mcmc_sampler_res(productions,
                                                             generalizations,
                                                             Dwin=Dwin,feat_probs=feat_probs,

                                                             ground_truth=eval(main_data_formatted['prior_partner'][i]),
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
               # post all (just based on all data points)
               for sample in range(n_1):
                    df_post_all = df_post_all.append(mh_mcmc_sampler_res(productions,
                                                           full_data,feat_probs=feat_probs,
                                                                      Dwin=Dwin,
                                                           ground_truth=eval(main_data_formatted['prior_partner'][i]),
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



                    global X                # defining a global variable X
                    X = []                  # for each scene, X will include the objects (i.e., cones) of the scene

                    # looping over the number of objects (ie cones) in each scene
                    for i_3 in range(0, len(gen['ids'])):
                         contact = check_structure(gen['contact'], i_3)  # converting misrepresented contact dictionaries into lists (see transform_functions.py for details)
                         # getting the properties for each object (triangle / cone) in the scene
                         object = {"id": gen['ids'][i_3], "colour":  gen['colours'][i_3] , "size":  gen['sizes'][i_3], "xpos":  int(np.round(gen['xpos'][i_3])),
                         "ypos":  int(np.round(gen['ypos'][i_3])), "rotation": np.round(gen['rotations'][i_3],1), "orientation": compute_orientation(gen['rotations'][i_3])[0],
                         "grounded":  gen['grounded'][i_3], "contact":  contact}

                         X.append(object)   # appending the object to X which includes all objects for a scene


                    # evaluating all sampled rules against the scenes for each of the different mcmc chains and appending results
                    for rule in df_prior['rulestring']:
                         res_prior.append(eval(rule))
                         # print(eval(rule))


               #      print(eval(max_a_posteriori_prior))
               #
               #      # print(eval("ob.forall(lambda x1: ob.or_operator(ob.equal(x1, 'blue','colour'), ob.equal(x1, 1,'size')), X)"))
               # print(max_a_posteriori_prior)
               #
               #      print(eval("ob.atleast(lambda x1: ob.and_operator(ob.equal(x1, 'blue','colour'), ob.equal(x1, 1,'size')), 1, X)"))

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
               raw_probs_all_trials_post_map.append(prob_gen_follow_rule_post_map)
               post_resp_all_trials_post_map.append(posterior_response)
               fitted_tau_post_map = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(prob_gen_follow_rule_post_map,posterior_response), method='L-BFGS-B')
               fitted_results_mod_post_map = fitted_probs(fitted_tau_post_map.x[0], prob_gen_follow_rule_post_map, posterior_response)
               select_probs_post_map.append(fitted_results_mod_post_map[0])


               # # and post all chain
               raw_probs_all_trials_post_all.append(prob_gen_follow_rule_post_all)
               post_resp_all_trials_post_all.append(posterior_response)
               fitted_tau_post_all = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(prob_gen_follow_rule_post_all,posterior_response), method='L-BFGS-B')
               fitted_results_mod_post_all = fitted_probs(fitted_tau_post_all.x[0], prob_gen_follow_rule_post_all, posterior_response)
               select_probs_post_all.append(fitted_results_mod_post_all[0])

               # # computing results of baseline model prior for a single trial
               baseline_probs = [.5, .5, .5, .5, .5, .5, .5, .5]
               fitted_results_baseline = fitted_probs(1, baseline_probs, prior_response)
               ll_baseline_prior.append(fitted_results_baseline[1])
               BICs_baseline_prior.append(-2 * fitted_results_baseline[1] + 0 * np.log(8))

               ####################### ALL TRIALS FOR ONE SUBJECT ##########################################
               if len(fitted_taus_prior) % 5 == 0:
                    raw_probs_all_trials_1 = [prob for sublist in raw_probs_all_trials_prior[i-4:] for prob in sublist]
                    prior_resp_all_trials_1 = [response for sublist in prior_resp_all_trials_prior[i-4:] for response in sublist]

                    overall_fitted_tau = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_trials_1,prior_resp_all_trials_1), method='L-BFGS-B')
                    fitted_results_all_trials_mod = fitted_probs(overall_fitted_tau.x[0], raw_probs_all_trials_1, prior_resp_all_trials_1)

                    baseline_probs_all_trials = [.5] * 40
                    fitted_results_all_trials_baseline = fitted_probs(1, baseline_probs_all_trials, prior_resp_all_trials_1)

                    raw_probs_all_trials_post_map_1 = [prob for sublist in raw_probs_all_trials_post_map[i-4:] for prob in sublist]
                    post_resp_all_trials_post_map_1 = [response for sublist in post_resp_all_trials_post_map[i-4:] for response in sublist]

                    overall_fitted_tau_post_map = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_trials_post_map_1, post_resp_all_trials_post_map_1), method='L-BFGS-B')
                    fitted_results_all_trials_mod_post_map = fitted_probs(overall_fitted_tau_post_map.x[0], raw_probs_all_trials_post_map_1 ,  post_resp_all_trials_post_map_1)

                    raw_probs_all_trials_post_all_1 = [prob for sublist in raw_probs_all_trials_post_all[i-4:] for prob in sublist]
                    post_resp_all_trials_post_all_1 = [response for sublist in post_resp_all_trials_post_all[i-4:] for response in sublist]

                    overall_fitted_tau_post_all = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_trials_post_all_1, post_resp_all_trials_post_all_1), method='L-BFGS-B')
                    fitted_results_all_trials_mod_post_all = fitted_probs(overall_fitted_tau_post_all.x[0], raw_probs_all_trials_post_all_1 ,  post_resp_all_trials_post_all_1)


                    baseline_probs_all_trials = [.5] * 40
                    fitted_results_all_trials_baseline_post_map = fitted_probs(1, baseline_probs_all_trials, post_resp_all_trials_post_map_1)
                    fitted_results_all_trials_baseline_post_all = fitted_probs(1, baseline_probs_all_trials, post_resp_all_trials_post_all_1)

                    #
                    prior_label = compare_BIC(fitted_results_all_trials_mod[1], fitted_results_all_trials_baseline[1], 40)
                    post_map_label = compare_BIC(fitted_results_all_trials_mod_post_map[1], fitted_results_all_trials_baseline_post_map[1], 40)
                    post_all_label = compare_BIC(fitted_results_all_trials_mod_post_all[1], fitted_results_all_trials_baseline_post_all[1], 40)



                    for trial in range(5):
                         fitted_taus_all_trials_prior.append(overall_fitted_tau.x[0])
                         select_probs_all_trials_prior.append(fitted_results_all_trials_mod[0])
                         ll_model_all_trials_prior.append(fitted_results_all_trials_mod[1])
                         BICs_model_all_trials_prior.append(-2 * fitted_results_all_trials_mod[1] + 1 * np.log(40))
                         ll_baseline_all_trials_prior.append(fitted_results_all_trials_baseline[1])
                         BICs_baseline_all_trials_prior.append(-2 * fitted_results_all_trials_baseline[1] + 0 * np.log(40))
                         raw_probs_all_trials_one_list_prior.append(raw_probs_all_trials_prior)

                         #
                         fitted_taus_all_trials_post_map.append(overall_fitted_tau_post_map.x[0])
                         select_probs_all_trials_post_map.append(fitted_results_all_trials_mod_post_map[0])
                         ll_model_all_trials_post_map.append(fitted_results_all_trials_mod_post_map[1])
                         BICs_model_all_trials_post_map.append(-2 * fitted_results_all_trials_mod_post_map[1] + 1 * np.log(40))
                         ll_baseline_all_trials_post_map.append(fitted_results_all_trials_baseline_post_map[1])
                         BICs_baseline_all_trials_post_map.append(-2 * fitted_results_all_trials_baseline_post_map[1] + 0 * np.log(40))
                         raw_probs_all_trials_one_list_post_map.append(raw_probs_all_trials_post_map)



                         fitted_taus_all_trials_post_all.append(overall_fitted_tau_post_all.x[0])
                         select_probs_all_trials_post_all.append(fitted_results_all_trials_mod_post_all[0])
                         ll_model_all_trials_post_all.append(fitted_results_all_trials_mod_post_all[1])
                         BICs_model_all_trials_post_all.append(-2 * fitted_results_all_trials_mod_post_all[1] + 1 * np.log(40))
                         ll_baseline_all_trials_post_all.append(fitted_results_all_trials_baseline_post_map[1])
                         BICs_baseline_all_trials_post_all.append(-2 * fitted_results_all_trials_baseline_post_map[1] + 0 * np.log(40))
                         raw_probs_all_trials_one_list_post_all.append(raw_probs_all_trials_post_all)

                         prior_labels.append(prior_label)
                         post_map_labels.append(post_map_label)
                         post_all_labels.append(post_all_label)





                    # computing accuracy of model predictions using hard maximization for selection probs
                    gt_all = gt * 5 # ground truth

                    prior_mod_select = hard_max_selections(raw_probs_all_trials_1)
                    post_map_mod_select = hard_max_selections(raw_probs_all_trials_post_map_1)
                    post_all_mod_select = hard_max_selections(raw_probs_all_trials_post_all_1)


                    prior_acc = sum([a and b or not a and not b for a, b in zip(gt_all, prior_mod_select)]) / len(prior_mod_select)
                    post_map_acc = sum([a and b or not a and not b for a, b in zip(gt_all, post_map_mod_select)]) / len(post_map_mod_select)
                    post_all_acc = sum([a and b or not a and not b for a, b in zip(gt_all, post_all_mod_select)]) / len(post_all_mod_select)

                    for acc in range(5):
                         prior_accs.append(prior_acc)
                         post_map_accs.append(post_map_acc)
                         post_all_accs.append(post_all_acc)

                    # single accuracies for prior
                    prior_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, prior_mod_select[:8])]) / 8)
                    prior_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, prior_mod_select[8:16])]) / 8)
                    prior_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, prior_mod_select[16:24])]) / 8)
                    prior_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, prior_mod_select[24:32])]) / 8)
                    prior_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, prior_mod_select[32:])]) / 8)



               # # creating a unique csv for each subject for each trial to store the outcome of the mcmc chains
               # df_prior.to_csv("sampled_rules_10_05/"+token_id + "_" + rule_name + "_prior.csv")       # storing a separate df including rules for each participant
               # df_post_map.to_csv("sampled_rules_10_05/"+token_id + "_" + rule_name + "_post_map.csv")
               # df_post_all.to_csv("sampled_rules_10_05/"+token_id + "_" + rule_name + "_post_all.csv")

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



          # prior trial specific data
          main_data_formatted = main_data_formatted[:100]
          main_data_formatted['raw_probs_prior'] = raw_probs_prior
          main_data_formatted['fitted_tau_prior'] = fitted_taus_prior
          main_data_formatted['log_ll_model_prior'] = ll_model_prior
          main_data_formatted['log_ll_baseline_prior'] = ll_baseline_prior
          main_data_formatted['BIC_model_prior'] = BICs_model_prior
          main_data_formatted['BIC_baseline_prior'] = BICs_baseline_prior
          main_data_formatted['select_probs_fitted_tau_prior'] = select_probs_prior

          # data across trials
          # prior
          main_data_formatted['raw_probs_all_trials_prior'] = raw_probs_all_trials_one_list_prior
          main_data_formatted['fitted_tau_all_trials_prior'] = fitted_taus_all_trials_prior
          main_data_formatted['log_ll_model_all_trials_prior'] = ll_model_all_trials_prior
          main_data_formatted['log_ll_baseline_all_trials_prior'] = ll_baseline_all_trials_prior
          main_data_formatted['BIC_model_all_trials_prior'] = BICs_model_all_trials_prior
          main_data_formatted['BIC_baseline_all_trials_prior'] = BICs_baseline_all_trials_prior
          main_data_formatted['select_probs_fitted_tau_all_trials_prior'] = select_probs_all_trials_prior

          # post map
          main_data_formatted['raw_probs_all_trials_post_map'] = raw_probs_all_trials_one_list_post_map
          main_data_formatted['fitted_tau_all_trials_post_map'] = fitted_taus_all_trials_post_map
          main_data_formatted['log_ll_model_all_trials_post_map'] = ll_model_all_trials_post_map
          main_data_formatted['log_ll_baseline_all_trials_post_map'] = ll_baseline_all_trials_post_map
          main_data_formatted['BIC_model_all_trials_post_map'] = BICs_model_all_trials_post_map
          main_data_formatted['BIC_baseline_all_trials_post_map'] = BICs_baseline_all_trials_post_map
          main_data_formatted['select_probs_fitted_tau_all_trials_post_map'] = select_probs_post_map
          #
          # # post all
          main_data_formatted['raw_probs_all_trials_post_all'] = raw_probs_all_trials_one_list_post_all
          main_data_formatted['fitted_tau_all_trials_post_all'] = fitted_taus_all_trials_post_all
          main_data_formatted['log_ll_model_all_trials_post_all'] = ll_model_all_trials_post_all
          main_data_formatted['log_ll_baseline_all_trials_post_all'] = ll_baseline_all_trials_post_all
          main_data_formatted['BIC_model_all_trials_post_all'] = BICs_model_all_trials_post_all
          main_data_formatted['BIC_baseline_all_trials_post_all'] = BICs_baseline_all_trials_post_all
          main_data_formatted['select_probs_fitted_tau_all_trials_post_all'] = select_probs_post_all

          # labels comparing baseline performance with mcmc performance (1 = mcmc wins for this subject)
          main_data_formatted['prior_label'] = prior_labels
          main_data_formatted['post_map_label'] = post_map_labels
          main_data_formatted['post_all_label'] = post_all_labels

          # accuracy of mcmc model predictions
          main_data_formatted['prior_acc'] = prior_accs
          main_data_formatted['prior_acc_single'] = prior_accs_single
          main_data_formatted['post_map_acc'] = post_map_accs
          main_data_formatted['post_all_acc'] = post_all_accs


          # maps
          main_data_formatted['map_prior'] = map_prior
          main_data_formatted['map_post_map'] = map_post_map
          main_data_formatted['map_post_all'] = map_post_all

          # map accs
          main_data_formatted['map_prior_acc'] = map_prior_acc
          main_data_formatted['map_post_map_acc'] = map_post_map_acc
          main_data_formatted['map_post_all_acc'] = map_post_all_acc



          main_data_formatted.to_csv('summary_adaptive_pruned_grammar_10_remain_val_probs' + str(rep) + '.csv')   # writing main data to new csv file including all relevant data for analysis

          rep+=1

predicted_selections(main_data_formatted)






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
