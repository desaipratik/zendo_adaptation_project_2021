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
from mcmc_connd_1_new_test_change_taut import mcmc_sampler, mcmc_sampler_map_surgery, mcmc_sampler_map
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

# split
main_data_formatted['change'] = [0] * sum(list(trial_counts.values()))
print(main_data_formatted['change'])
change_ind = 0
for change in main_data_formatted['change']:
    init_rule = main_data_formatted['prior_resp'][change_ind]
    rev_rule = main_data_formatted['post_resp'][change_ind]
    # int_rule = rr.list_to_string(rr.string_to_list(init_rule))
    init_rule_tight = rr.list_to_string(rr.string_to_list(init_rule))
    rev_rule_tight = rr.list_to_string(rr.string_to_list(rev_rule))
    # rev_rule = rr.list_to_string(rr.string_to_list(rev_rule))

    if init_rule_tight != rev_rule_tight:
        main_data_formatted['change'][change_ind] = 1


    change_ind+=1


main_data_formatted = main_data_formatted.query("change == 1")
main_data_formatted = main_data_formatted.reset_index(drop=True)
trial_counts = main_data_formatted.groupby('token_id').size()
trial_counts = dict(trial_counts)
print(len(main_data_formatted['change']))

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


          n_rows =42
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
               bv_prior = eval(main_data_formatted['bound_vars'][i])
               subj_rule = rr.list_to_string(rr.string_to_list(subj_rule))

               print(subj_rule)
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
               full_data = init_trials + rev_trials # full data used for the third mcmc chain combining trials and generalizations


               # getting additional training data for evaluation of model
               training_data = rand_scene_creator(correct_rule, n_scenes=0)
               test_training_data = rand_scene_creator(correct_rule, n_scenes=100)
               full_training_dat_prior = init_trials + training_data
               label_dat_post_map = generalizations
               full_training_dat_post_map = rev_trials + label_dat_post_map
               full_training_dat_post_all = full_data + training_data
               full_training_dat_post_all_prior = rev_trials + init_trials


               prior_probs = get_production_probs_prototype(full_training_dat_prior,'prior',cond='1',feat_only=False)
               Dwin_prior = prior_probs[0]
               feat_probs_prior = prior_probs[1]

               # #same for both post map and post all
               post_probs = get_production_probs_prototype(full_training_dat_post_all ,'post',cond='1', feat_only=False)
               Dwin_post = post_probs[0]
               feat_probs_post = post_probs[1]

               # print(main_data_formatted['rule_description'][i])

               # prior
               df_prior = mcmc_sampler(productions,replacements, full_training_dat_prior,Dwin=Dwin_prior,ground_truth_labels=False, iter = 0)[0] # creating empty data for first initial rules
               prior_details = [] # details for prior rules are needed to run the map mcmc chain which starts with the map rule of the prior

               df_prior_labels = mcmc_sampler(productions,replacements, label_dat_post_map,Dwin=Dwin_prior,ground_truth_labels=False, iter = 0)[0] # creating empty data for first initial rules
               prior_details_labels = [] # details for prior rules are needed to run the map mcmc chain which starts with the map rule of the prior

               # post map chain
               df_post_map = pd.DataFrame({"rulestring": []})# creating empty data for revised rules

               # post all chain
               df_post_all = mcmc_sampler(productions,replacements, full_training_dat_post_all, Dwin=Dwin,iter = 0)[0] # creating empty data for all rules

                # post all chain
               df_post_all_seed = mcmc_sampler(productions,replacements, full_training_dat_post_all, Dwin=Dwin,iter = 0)[0] # creating empty data for all rules


               # mcmc chains
               # prior
               # print('prior_scenes')
               for sample in range(n_1):
                    prior_df_and_details = mcmc_sampler(productions,
                                                               replacements,
                                                               full_training_dat_prior,
                                                               # iter_1 = n_1,
                                                               Dwin=Dwin_prior,
                                                               feat_probs=feat_probs_prior,
                                                               ground_truth_labels=False,
                                                               iter = n_2,
                                                               type='prior',
                                                               out_penalizer=2,
                                                               row=len(df_prior['rulestring']))
                    df_prior = df_prior.append(prior_df_and_details[0])
                    prior_details.append(prior_df_and_details[1])
               #convert list of dicts into single dict
               prior_detail = {}
               # for dict in prior_details:
               # print(df_prior['rulestring'])
               max_a_posteriori_prior = df_prior['rulestring'].mode()[0]     # getting the mode of the prior chain first to initialise the post map chain
               map_index = np.where(df_prior["rulestring"] == max_a_posteriori_prior)[0][0]
               map_prior.append(max_a_posteriori_prior)

               prior_rule_strings = df_prior['rulestring'].tolist()
               rules_prior.append(prior_rule_strings)
               correct_rule_perc_prior.append(prior_rule_strings.count(rule_string))
               # correct_rule_perc_prior.append(len(prior_rule_strings))
               # print(correct_rule_perc_prior)

               # print('prior_labesl')
               for sample in range(1):
                    wsls_check = mcmc_sampler(productions,
                                                                     replacements,
                                                               rev_trials,
                                                              bv=bv_prior,
                                                              start=subj_rule,
                                                               ground_truth_labels=prior_response,

                                                               Dwin=Dwin_prior,
                                                               feat_probs=feat_probs_prior,
                                                               iter = 1,
                                                               type='prior_label',
                                                               out_penalizer=4,
                                                               row=len(df_prior_labels['rulestring']))
                    # df_prior_labels = df_prior_labels.append(prior_label_df_and_details[0])
                    # prior_details_labels.append(prior_label_df_and_details[1])

               #convert list of dicts into single dict
               # prior_detail_label = {}
               # for dict in prior_details_labels:
               #      prior_detail_label.update(dict)
               max_a_posteriori_prior_label = rule_string     # getting the mode of the prior chain first to initialise the post map chain
               # # map_index_label = np.where(df_prior_labels["rulestring"] == max_a_posteriori_prior_label)[0][0]
               # map_prior.append(max_a_posteriori_prior)
               # print(df_prior_labels['rulestring'].mode()[0])


               prior_rule_strings_label = [subj_rule] * 1000
               rules_prior_label.append(prior_rule_strings_label)
               correct_rule_perc_prior_labels.append(prior_rule_strings_label.count(rule_string))
               # print(prior_rule_strings_label)
               print(correct_rule_perc_prior_labels)
               print('mewo')

               # computing the accuracy of the other player
               # acc_partner = sum([a and b or not a and not b for a,b in zip(gt, eval(main_data_formatted['prior_partner'][i]))]) / 8
#                # print('post_map')

#                # print(df_prior_labels['rulestring'])

#                # print(df_prior['rulestring'])
               row_count = 0
#                # print(df_prior['rulestring'])
               for sample in range(1):
#                     # print('cat')
#                     # print('qhale')
#                     # print(df_prior['rulestring'].mode()[0])
#                     # print('giant_tendaaaai')
#                     # print(df_prior_labels['rulestring'].mode()[0])
                    map_index_label = row_count
                         # np.random.randint(len(df_prior_labels['rulestring']))
#                     # print(df_prior_labels['rulestring'][map_index_label])
                         # np.random.randint(len(df_prior_labels['rulestring']))
                         # np.where(df_prior_labels["rulestring"] == max_a_posteriori_prior_label)[0][0]
                         # np.random.randint(len(df_prior_labels['rulestring']))
#                     # print(map_index_label)
#                     # print(len(df_prior_labels['rulestring']))
#                     # print(df_prior_labels['rulestring'][map_index_label])
#                     # print(df_post_map)
#                     # print('intermediate postmap above')
                    df_post_map = df_post_map.dropna()
#                     # print(df_post_map)
#                     # print('dropped abvoe ')
                    df_post_map = df_post_map.append(mcmc_sampler_map_surgery(productions,replacements,
                                                             full_training_dat_post_map,
                                                             Dwin=Dwin_post,
                                                                              test_data = test_training_data,
                                                             # iter_1 = n_1,
                                                                             ground_truth_labels=prior_response,
                                                                         feat_probs=feat_probs_post,
                                                             start= subj_rule,
                                                                      # df_prior_labels['rulestring'][map_index_label],
                                                                         # df_prior_labels['rulestring'].mode()[0] ,


                                                             iter = 1000,
                                                             # type='posterior_map',
                                                                      bv = bv_prior,
                                                             # df_prior_labels['bv'][map_index_label],

                                                                         row=row_count,
                                                             out_penalizer=4)[0])
                    row_count+=1


               # print('now rulestring below')
               # print(df_post_map['rulestring'])

               #

               acceptance_probs = df_post_map['acceptance_probs'].tolist()
               max_ind = acceptance_probs.index(max(acceptance_probs))
               df_post_map['rulestring'] = df_post_map['rulestring'][max_ind]
               print(df_post_map['rulestring'])

#                # print(df_post_map['rulestring'])
               max_a_posteriori_post_map = df_post_map['rulestring'].mode()[0]
               map_post_map.append(max_a_posteriori_post_map)

               post_map_rule_strings = df_post_map['rulestring'].tolist()
               rules_post_map.append(post_map_rule_strings)
               # print(post_map_rule_strings)
               print('cor')
               correct_rule_perc_post_map.append(post_map_rule_strings.count(rule_string))
               print(correct_rule_perc_post_map)
               print('mewomewo')
               # print(correct_rule_perc_post_map)
               print('postmaprulestring')
               print(df_post_map["rulestring"])
               #
               # print('post_all_seed')
               # post all (just based on all data points)
               if (1 - wsls_check) > np.random.rand():
                   print(np.random.rand())
                   print(wsls_check)
                   print('resampling')
                   for sample in range(n_1):
                        df_post_all = df_post_all.append(mcmc_sampler(productions,
                                                                             replacements,
                                                               full_training_dat_post_all,
                                                               Dwin=Dwin_post,
                                                                             feat_probs=feat_probs_post,
                                                                      start='S',
                                                               iter = 10,


                                                               type='posterior_all', row=len(df_post_all['rulestring']),
                                                               out_penalizer=2)[0])

               else:
                   df_post_all = pd.DataFrame({'rulestring': [subj_rule] * 1000})

               max_a_posteriori_post_all = df_post_all['rulestring'].mode()[0]
               map_post_all.append(max_a_posteriori_post_all)
#                # print(df_post_all)
               post_all_rule_strings = df_post_all['rulestring'].tolist()
               correct_rule_perc_post_all.append(post_all_rule_strings.count(rule_string))
               print(correct_rule_perc_post_all)
               rules_post_all.append(post_all_rule_strings)






               # print('post_all_seed')
               # post all (just based on all data points)
               for sample in range(n_1):
                    df_post_all_seed = df_post_all_seed.append(mcmc_sampler_map(productions,
                                                                         replacements,full_training_dat_post_map,
                                                                            start=subj_rule,
                                                                            bv=bv_prior,
                                                                            ground_truth_labels=prior_response,
                                                                                  test_data = test_training_data,

                                                           Dwin=Dwin_post,
                                                                         feat_probs=feat_probs_post,
                                                           iter = 1000,


                                                           row=len(df_post_all_seed['rulestring']),
                                                           out_penalizer=4)[0])


               #
               acceptance_probs = df_post_all_seed['acceptance_probs'].tolist()
               max_ind = acceptance_probs.index(max(acceptance_probs))
               df_post_all_seed['rulestring'] = df_post_all_seed['rulestring'][max_ind]
               print(df_post_all_seed['rulestring'])

               max_a_posteriori_post_all_seed = df_post_all_seed['rulestring'].mode()[0]
               map_post_all_seed.append(max_a_posteriori_post_all_seed)
#                # print(df_post_all)
               post_all_rule_strings_seed = df_post_all_seed['rulestring'].tolist()
               rules_post_all_seed.append(post_all_rule_strings_seed)
               correct_rule_perc_post_all_seed.append(post_all_rule_strings_seed.count(rule_string))
               print(correct_rule_perc_post_all_seed)
               print('cor')

               # calculating the probability that a scene is rule following for a specific trial and subject
               prob_gen_follow_rule_prior = []
               prob_gen_follow_rule_prior_label = []
               prob_gen_follow_rule_post_map = []
               prob_gen_follow_rule_post_all = []
               prob_gen_follow_rule_post_all_seed = []

               map_res_prior = []                # result checks whether this scene follows a rule (length of results equals n_1 * n_2)
               map_res_prior_label = []
               map_res_post_map = []
               map_res_post_all = []
               map_res_post_all_seed = []

               # evaluating the rules based on the generalization data shown to subjects
               for gen in generalizations:        # looping over all 8 generalization scenes


                    res_prior = []                # result checks whether this scene follows a rule (length of results equals n_1 * n_2)
                    res_prior_label = []
                    res_post_map = []
                    res_post_all = []
                    res_post_all_seed = []



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
#                     # print(Z.exists(lambda x1: Z.forall(lambda x2: Z.and_operator(Z.and_operator(Z.equal(x1,'red','colour'), Z.not_operator(Z.equal(x2, 'red', 'colour'))), Z.greater(x1,x2,'size')), X), X))
                    # evaluating all sampled rules against the scenes for each of the different mcmc chains and appending results
                    for rule in df_prior['rulestring']:
                         res_prior.append(eval(rule))
#                          # print(eval(rule))
#                     # print(X)
#                     # print(Z.forall(lambda x1: Z.not_operator(Z.equal(x1, 'upright', 'orientation')), X))
                    # for rule in df_prior_labels['rulestring']:
#                     #      print(subj_rule)
                    res_prior_label.append(eval(subj_rule))

                    for rule in df_post_map['rulestring']:
#                          # print(rule)
                         res_post_map.append(eval(rule))

                    for rule in df_post_all['rulestring']:
                         res_post_all.append(eval(rule))

                    for rule in df_post_all_seed['rulestring']:
                         res_post_all_seed.append(eval(rule))

                    map_res_prior.append(eval(max_a_posteriori_prior))
                    map_res_prior_label.append(eval(max_a_posteriori_prior_label))
                    map_res_post_map.append(eval(max_a_posteriori_post_map))
                    map_res_post_all.append(eval(max_a_posteriori_post_all))
                    map_res_post_all_seed.append(eval(max_a_posteriori_post_all_seed))

                    # computing the raw probabilities that the scenes follow a rule for each chain

                    p_follow_rule_prior = (1 / len(res_prior)) * sum(res_prior) # len(res) = number of rules; sum(res) = number of rules matching the scene
#                     # print(p_follow_rule_prior)
                    prob_gen_follow_rule_prior.append(p_follow_rule_prior)

                    p_follow_rule_prior_label = (1 / len(res_prior_label)) * sum(res_prior_label) # len(res) = number of rules; sum(res) = number of rules matching the scene
#                     # print(p_follow_rule_prior)
                    print(len(res_prior_label))
                    print(sum(res_prior_label))
                    prob_gen_follow_rule_prior_label.append(p_follow_rule_prior_label)





                    p_follow_rule_post_map = (1 / len(res_post_map)) * sum(res_post_map) # len(res) = number of rules; sum(res) = number of rules matching the scene
                    prob_gen_follow_rule_post_map.append(p_follow_rule_post_map)

                    p_follow_rule_post_all = (1 / len(res_post_all)) * sum(res_post_all) # len(res) = number of rules; sum(res) = number of rules matching the scene
                    prob_gen_follow_rule_post_all.append(p_follow_rule_post_all)

                    p_follow_rule_post_all_seed = (1 / len(res_post_all_seed)) * sum(res_post_all_seed) # len(res) = number of rules; sum(res) = number of rules matching the scene
                    prob_gen_follow_rule_post_all_seed.append(p_follow_rule_post_all_seed)

               map_prior_acc.append(compute_acc(gt, map_res_prior))
               map_prior_acc_label.append(compute_acc(gt, map_res_prior_label))
               map_post_map_acc.append(compute_acc(gt, map_res_post_map))
               map_post_all_acc.append(compute_acc(gt, map_res_post_all))
               map_post_all_acc_seed.append(compute_acc(gt, map_res_post_all_seed))


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


               # prior label chains

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

                # # and post all chain seed
               raw_probs_post_all_seed.append(prob_gen_follow_rule_post_all_seed)
               raw_probs_all_trials_post_all_seed.append(prob_gen_follow_rule_post_all_seed)
               post_resp_all_trials_post_all_seed.append(posterior_response)

               fitted_tau_post_all_seed = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(prob_gen_follow_rule_post_all_seed,posterior_response), method='L-BFGS-B')
               fitted_taus_post_all_seed.append(fitted_tau_post_all_seed.x[0])
               fitted_results_mod_post_all_seed = fitted_probs(fitted_tau_post_all_seed.x[0], prob_gen_follow_rule_post_all_seed, posterior_response)
               select_probs_post_all_seed.append(fitted_results_mod_post_all_seed[0])
               ll_model_post_all_seed.append(fitted_results_mod_post_all_seed[1])
               BICs_model_post_all_seed.append(-2 * fitted_results_mod_post_all_seed[1] + 1 * np.log(8))

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
               # print(len(fitted_taus_prior))
               # print(n_trials)
               # print('ntrialsabove')
               if n_trials_counter % n_trials == 0:
                    n_trials_counter = 0
                    # print(n_trials_counter)
                    # print('hha')
                    # print(n_trials)
                    # # print('hi')
                    # # print(n_trials)
                    raw_probs_all_trials_1 = [prob for sublist in raw_probs_all_trials_prior[i-n_trials+1:] for prob in sublist]
                    prior_resp_all_trials_1 = [response for sublist in prior_resp_all_trials_prior[i-n_trials+1:] for response in sublist]


                    overall_fitted_tau = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_trials_1,prior_resp_all_trials_1), method='L-BFGS-B')
                    fitted_results_all_trials_mod = fitted_probs(overall_fitted_tau.x[0], raw_probs_all_trials_1, prior_resp_all_trials_1)

                    raw_probs_all_trials_label_1 = [prob for sublist in raw_probs_all_trials_prior_labels[i-n_trials+1:] for prob in sublist]
                    prior_resp_all_trials_label_1 = [response for sublist in prior_resp_all_trials_prior_labels[i-n_trials+1:] for response in sublist]


                    overall_fitted_tau_label = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_trials_label_1,prior_resp_all_trials_label_1), method='L-BFGS-B')
                    fitted_results_all_trials_mod_label = fitted_probs(overall_fitted_tau_label.x[0], raw_probs_all_trials_label_1, prior_resp_all_trials_label_1)



                    baseline_probs_all_trials = [.5] * 8 * n_trials
                    fitted_results_all_trials_baseline = fitted_probs(1, baseline_probs_all_trials, prior_resp_all_trials_1)

                    raw_probs_all_trials_post_map_1 = [prob for sublist in raw_probs_all_trials_post_map[i-n_trials+1:] for prob in sublist]
                    post_resp_all_trials_post_map_1 = [response for sublist in post_resp_all_trials_post_map[i-n_trials+1:] for response in sublist]

                    overall_fitted_tau_post_map = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_trials_post_map_1, post_resp_all_trials_post_map_1), method='L-BFGS-B')
                    fitted_results_all_trials_mod_post_map = fitted_probs(overall_fitted_tau_post_map.x[0], raw_probs_all_trials_post_map_1 ,  post_resp_all_trials_post_map_1)

                    raw_probs_all_trials_post_all_1 = [prob for sublist in raw_probs_all_trials_post_all[i-n_trials+1:] for prob in sublist]
                    post_resp_all_trials_post_all_1 = [response for sublist in post_resp_all_trials_post_all[i-n_trials+1:] for response in sublist]

                    overall_fitted_tau_post_all = minimize(get_tau, 1, bounds=[(0.1, 100.00)],args=(raw_probs_all_trials_post_all_1, post_resp_all_trials_post_all_1), method='L-BFGS-B')
                    fitted_results_all_trials_mod_post_all = fitted_probs(overall_fitted_tau_post_all.x[0], raw_probs_all_trials_post_all_1 ,  post_resp_all_trials_post_all_1)

                    raw_probs_all_trials_post_all_1_seed = [prob for sublist in raw_probs_all_trials_post_all_seed[i-n_trials+1:] for prob in sublist]
                    post_resp_all_trials_post_all_1_seed = [response for sublist in post_resp_all_trials_post_all_seed[i-n_trials+1:] for response in sublist]

                    overall_fitted_tau_post_all_seed = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_trials_post_all_1_seed, post_resp_all_trials_post_all_1_seed), method='L-BFGS-B')
                    fitted_results_all_trials_mod_post_all_seed = fitted_probs(overall_fitted_tau_post_all_seed.x[0], raw_probs_all_trials_post_all_1_seed, post_resp_all_trials_post_all_1_seed)




                    baseline_probs_all_trials = [.5] * 8 * n_trials
                    fitted_results_all_trials_baseline_post_map = fitted_probs(1, baseline_probs_all_trials, post_resp_all_trials_post_map_1)
                    fitted_results_all_trials_baseline_post_all = fitted_probs(1, baseline_probs_all_trials, post_resp_all_trials_post_all_1)

                    #
                    prior_label = compare_BIC(fitted_results_all_trials_mod[1], fitted_results_all_trials_baseline[1], 8*n_trials)
                    prior_label_label = compare_BIC(fitted_results_all_trials_mod_label[1], fitted_results_all_trials_baseline[1], 8*n_trials)
                    post_map_label = compare_BIC(fitted_results_all_trials_mod_post_map[1], fitted_results_all_trials_baseline_post_map[1], 8*n_trials)
                    post_all_label = compare_BIC(fitted_results_all_trials_mod_post_all[1], fitted_results_all_trials_baseline_post_all[1], 8*n_trials)
                    post_all_label_seed = compare_BIC(fitted_results_all_trials_mod_post_all_seed[1], fitted_results_all_trials_baseline_post_all[1], 8*n_trials)


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

                         fitted_tau_all_subjects_prior = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_subjects,prior_resp_all_subjects), method='L-BFGS-B')
#                          # print(fitted_taus_all_subjects_prior)
                         fitted_results_all_subjects_prior = fitted_probs(fitted_tau_all_subjects_prior.x[0], raw_probs_all_subjects,prior_resp_all_subjects)


                         fitted_taus_all_subjects_prior = []
                         fitted_taus_all_subjects_prior.append(float(fitted_tau_all_subjects_prior.x[0]))

                         select_probs_all_subjects_prior = []
                         select_probs_all_subjects_prior.append(fitted_results_all_subjects_prior[0])

                         ll_model_all_subjects_prior = []
                         ll_model_all_subjects_prior.append(fitted_results_all_subjects_prior[1])

                         BICs_model_all_subjects_prior = []
                         BICs_model_all_subjects_prior.append(-2 * fitted_results_all_subjects_prior[1] + 1 * np.log(8 * len(main_data_formatted['rule_name'][:n_rows])))

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

                         raw_probs_all_subjects_post_map = [prob for sublist in raw_probs_all_trials_post_map for prob in sublist]
                         resp_all_subjects_post_map = [response for sublist in post_resp_all_trials_post_map for response in sublist]

                         fitted_tau_all_subjects_post_map = minimize(get_tau, 1, bounds=[(0.1, 100.00)],args=(raw_probs_all_subjects_post_map,resp_all_subjects_post_map), method='L-BFGS-B')
                         fitted_results_all_subjects_post_map = fitted_probs(fitted_tau_all_subjects_post_map.x[0], raw_probs_all_subjects_post_map,resp_all_subjects_post_map)

                         fitted_taus_all_subjects_post_map = []
                         fitted_taus_all_subjects_post_map.append(float(fitted_tau_all_subjects_post_map.x[0]))

                         select_probs_all_subjects_post_map = []
                         select_probs_all_subjects_post_map.append(fitted_results_all_subjects_post_map[0])

                         ll_model_all_subjects_post_map = []
                         ll_model_all_subjects_post_map.append(fitted_results_all_subjects_post_map[1])

                         BICs_model_all_subjects_post_map = []
                         BICs_model_all_subjects_post_map.append(-2 * fitted_results_all_subjects_post_map[1] + 1 * np.log(8 * len(main_data_formatted['rule_name'][:n_rows])))



                         # post all

                         raw_probs_all_subjects_post_all = [prob for sublist in raw_probs_all_trials_post_all for prob in sublist]
                         print(raw_probs_all_subjects_post_all)
                         resp_all_subjects_post_all = [response for sublist in post_resp_all_trials_post_all for response in sublist]

                         fitted_tau_all_subjects_post_all = minimize(get_tau, 1, bounds=[(0.01, 10000.00)],args=(raw_probs_all_subjects_post_all,resp_all_subjects_post_all), method='L-BFGS-B')
                         fitted_results_all_subjects_post_all = fitted_probs(fitted_tau_all_subjects_post_all.x[0], raw_probs_all_subjects_post_all,resp_all_subjects_post_all)

                         fitted_taus_all_subjects_post_all = []
                         fitted_taus_all_subjects_post_all.append(float(fitted_tau_all_subjects_post_all.x[0]))

                         select_probs_all_subjects_post_all = []
                         select_probs_all_subjects_post_all.append(fitted_results_all_subjects_post_all[0])

                         ll_model_all_subjects_post_all = []
                         ll_model_all_subjects_post_all.append(fitted_results_all_subjects_post_all[1])

                         BICs_model_all_subjects_post_all = []
                         BICs_model_all_subjects_post_all.append(-2 * fitted_results_all_subjects_post_all[1] + 1 * np.log(8 * len(main_data_formatted['rule_name'][:n_rows])))

                          # post all seed

                         raw_probs_all_subjects_post_all_seed = [prob for sublist in raw_probs_all_trials_post_all_seed for prob in sublist]
                         resp_all_subjects_post_all_seed = [response for sublist in post_resp_all_trials_post_all_seed for response in sublist]

                         fitted_tau_all_subjects_post_all_seed = minimize(get_tau, 1, bounds=[(0.1, 100.00)],args=(raw_probs_all_subjects_post_all_seed,resp_all_subjects_post_all_seed), method='L-BFGS-B')
                         fitted_results_all_subjects_post_all_seed = fitted_probs(fitted_tau_all_subjects_post_all_seed.x[0], raw_probs_all_subjects_post_all_seed,resp_all_subjects_post_all_seed)

                         fitted_taus_all_subjects_post_all_seed = []
                         fitted_taus_all_subjects_post_all_seed.append(float(fitted_tau_all_subjects_post_all_seed.x[0]))

                         select_probs_all_subjects_post_all_seed = []
                         select_probs_all_subjects_post_all_seed.append(fitted_results_all_subjects_post_all_seed[0])

                         ll_model_all_subjects_post_all_seed = []
                         ll_model_all_subjects_post_all_seed.append(fitted_results_all_subjects_post_all_seed[1])

                         BICs_model_all_subjects_post_all_seed = []
                         BICs_model_all_subjects_post_all_seed.append(-2 * fitted_results_all_subjects_post_all_seed[1] + 1 * np.log(8 * len(main_data_formatted['rule_name'][:n_rows])))


                         # baseline

                         baseline_probs_all_subjects = [.5] * 8 * len(main_data_formatted['rule_name'])
                         fitted_results_all_subjects_baseline = fitted_probs(1, baseline_probs_all_subjects, prior_resp_all_subjects)


                         ll_baseline_all_subjects_prior = []
                         #ll_baseline_all_subjects_prior.append(fitted_results_all_subjects_baseline[1])

                         ll_baseline_all_subjects_prior.append(fitted_results_all_subjects_baseline[1])

                         BICs_baseline_all_subjects_prior = []
                         BICs_baseline_all_subjects_prior.append(-2 * fitted_results_all_subjects_baseline[1])
#                          # print(BICs_baseline_all_subjects_prior)

                         # BICs_baseline_all_subjects_prior = []
                         # BICs_baseline_all_subjects_prior.append(-2 * fitted_results_all_subjects_baseline[1] + 0 * np.log(40 * len(fitted_taus_prior)))

                         # overall accuracy
                         gt_all_subjects = gt * len(fitted_taus_prior) # ground truth

                         prior_mod_select_all_subj = hard_max_selections(raw_probs_all_subjects)
                         prior_mod_select_all_subj_label = hard_max_selections(raw_probs_all_subjects_label)
                         post_map_mod_select_all_subj = hard_max_selections(raw_probs_all_subjects_post_map)
                         post_all_mod_select_all_subj = hard_max_selections(raw_probs_all_subjects_post_all)
                         post_all_mod_select_all_subj_seed = hard_max_selections(raw_probs_all_subjects_post_all_seed)


                         prior_acc_all_subj = sum([a and b or not a and not b for a, b in zip(gt_all_subjects, prior_mod_select_all_subj)]) / len(prior_mod_select_all_subj)
                         prior_acc_all_subj_label = sum([a and b or not a and not b for a, b in zip(gt_all_subjects, prior_mod_select_all_subj_label)]) / len(prior_mod_select_all_subj_label)
                         post_map_acc_all_subj = sum([a and b or not a and not b for a, b in zip(gt_all_subjects, post_map_mod_select_all_subj)]) / len(post_map_mod_select_all_subj)
                         post_all_acc_all_subj = sum([a and b or not a and not b for a, b in zip(gt_all_subjects, post_all_mod_select_all_subj)]) / len(post_all_mod_select_all_subj)
                         post_all_acc_all_subj_seed = sum([a and b or not a and not b for a, b in zip(gt_all_subjects, post_all_mod_select_all_subj_seed)]) / len(post_all_mod_select_all_subj_seed)



#                          # print(len(fitted_taus_prior))








                    for trial in range(n_trials):

                         fitted_taus_all_trials_prior.append(overall_fitted_tau.x[0])
                         select_probs_all_trials_prior.append(fitted_results_all_trials_mod[0])
                         ll_model_all_trials_prior.append(fitted_results_all_trials_mod[1])
                         BICs_model_all_trials_prior.append(-2 * fitted_results_all_trials_mod[1] + 1 * np.log(8*n_trials))
                         ll_baseline_all_trials_prior.append(fitted_results_all_trials_baseline[1])
                         BICs_baseline_all_trials_prior.append(-2 * fitted_results_all_trials_baseline[1] + 0 * np.log(8*n_trials))
                         raw_probs_all_trials_one_list_prior.append(raw_probs_all_trials_prior)

                         fitted_taus_all_trials_prior_labels.append(overall_fitted_tau_label.x[0])
                         select_probs_all_trials_prior_labels.append(fitted_results_all_trials_mod_label[0])
                         ll_model_all_trials_prior_labels.append(fitted_results_all_trials_mod_label[1])
                         BICs_model_all_trials_prior_labels.append(-2 * fitted_results_all_trials_mod_label[1] + 1 * np.log(8*n_trials))
                         raw_probs_all_trials_one_list_prior_labels.append(raw_probs_all_trials_prior_labels)


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

                         fitted_taus_all_trials_post_all_seed.append(overall_fitted_tau_post_all_seed.x[0])
                         select_probs_all_trials_post_all_seed.append(fitted_results_all_trials_mod_post_all_seed[0])
                         ll_model_all_trials_post_all_seed.append(fitted_results_all_trials_mod_post_all_seed[1])
                         BICs_model_all_trials_post_all_seed.append(-2 * fitted_results_all_trials_mod_post_all_seed[1] + 1 * np.log(8*n_trials))
                         ll_baseline_all_trials_post_all_seed.append(fitted_results_all_trials_baseline_post_map[1])
                         BICs_baseline_all_trials_post_all_seed.append(-2 * fitted_results_all_trials_baseline_post_map[1] + 0 * np.log(8*n_trials))
                         raw_probs_all_trials_one_list_post_all_seed.append(raw_probs_all_trials_post_all_seed)


                         prior_labels.append(prior_label)
                         prior_labels_labels.append(prior_label_label)
                         post_map_labels.append(post_map_label)
                         post_all_labels.append(post_all_label)
                         post_all_labels_seed.append(post_all_label_seed)






                    # computing accuracy of model predictions using hard maximization for selection probs
                    gt_all = gt * n_trials # ground truth

                    prior_mod_select = hard_max_selections(raw_probs_all_trials_1)
                    prior_mod_select_labels = hard_max_selections(raw_probs_all_trials_label_1)
                    post_map_mod_select = hard_max_selections(raw_probs_all_trials_post_map_1)
                    post_all_mod_select = hard_max_selections(raw_probs_all_trials_post_all_1)
                    post_all_mod_select_seed = hard_max_selections(raw_probs_all_trials_post_all_1_seed)

                    prior_acc = sum([a and b or not a and not b for a, b in zip(gt_all, prior_mod_select)]) / len(prior_mod_select)
                    prior_acc_label = sum([a and b or not a and not b for a, b in zip(gt_all, prior_mod_select_labels)]) / len(prior_mod_select_labels)
                    post_map_acc = sum([a and b or not a and not b for a, b in zip(gt_all, post_map_mod_select)]) / len(post_map_mod_select)
                    post_all_acc = sum([a and b or not a and not b for a, b in zip(gt_all, post_all_mod_select)]) / len(post_all_mod_select)
                    post_all_acc_seed = sum([a and b or not a and not b for a, b in zip(gt_all, post_all_mod_select_seed)]) / len(post_all_mod_select_seed)

                    low_bound = 0
                    up_bound = 8
                    for acc in range(n_trials):
                         prior_accs.append(prior_acc)
                         prior_label_accs.append(prior_acc_label)
                         post_map_accs.append(post_map_acc)
                         post_all_accs.append(post_all_acc)
                         post_all_accs_seed.append(post_all_acc_seed)
                         # single accuracies for prior
                         prior_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, prior_mod_select[low_bound:up_bound])]) / 8)
                         prior_label_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, prior_mod_select_labels[low_bound:up_bound])]) / 8)
                         post_map_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, post_map_mod_select[low_bound:up_bound])]) / 8)
                         post_all_accs_single.append(sum([a and b or not a and not b for a, b in zip(gt, post_all_mod_select[low_bound:up_bound])]) / 8)
                         post_all_accs_single_seed.append(sum([a and b or not a and not b for a, b in zip(gt, post_all_mod_select_seed[low_bound:up_bound])]) / 8)
                         low_bound+=8
                         up_bound+=8






               # creating a unique csv for each subject for each trial to store the outcome of the mcmc chains
               # df_prior.to_csv("model_results/"+token_id + "_" + rule_name + "_prior_cond_3.csv")       # storing a separate df including rules for each participant
               # df_post_map.to_csv("model_results/"+token_id + "_" + rule_name + "_post_map_cond_3.csv")
               # df_post_all.to_csv("model_results/"+token_id + "_" + rule_name + "_post_all_cond_3.csv")
#                # print(raw_probs_all_trials_one_list_prior)
               i+=1 # pr
               n_trials_counter+=1
               print(i)
               # oceeding to next trial
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


#           # print(raw_probs_all_trials_one_list_prior)
          # prior trial specific data

          main_data_formatted = main_data_formatted[:n_rows]

          # main_data_formatted['raw_probs_prior'] = raw_probs_prior
          main_data_formatted['fitted_tau_prior'] = fitted_taus_prior
          main_data_formatted['log_ll_model_prior'] = ll_model_prior
          main_data_formatted['log_ll_baseline_prior'] = ll_baseline_prior
          main_data_formatted['BIC_model_prior'] = BICs_model_prior
          main_data_formatted['BIC_baseline_prior'] = BICs_baseline_prior
          # main_data_formatted['select_probs_fitted_tau_prior'] = select_probs_prior

          # prior data across trials
          # main_data_formatted['raw_probs_all_trials_prior'] = raw_probs_all_trials_one_list_prior
          # print(fitted_taus_all_trials_prior)
          main_data_formatted['fitted_tau_all_trials_prior'] = fitted_taus_all_trials_prior
          main_data_formatted['log_ll_model_all_trials_prior'] = ll_model_all_trials_prior
          main_data_formatted['log_ll_baseline_all_trials_prior'] = ll_baseline_all_trials_prior
          main_data_formatted['BIC_model_all_trials_prior'] = BICs_model_all_trials_prior
          main_data_formatted['BIC_baseline_all_trials_prior'] = BICs_baseline_all_trials_prior
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
          main_data_formatted['fitted_taus_all_subjects_prior'] = fitted_taus_all_subjects_prior * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['raw_probs_all_subjects_prior'] = [raw_probs_all_subjects] * len(main_data_formatted['rule_name'][:n_rows])
#           # print(len(select_probs_all_subjects_prior * 450)
          main_data_formatted['ll_model_all_subjects_prior'] = ll_model_all_subjects_prior* len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['BICs_model_all_subjects_prior'] = BICs_model_all_subjects_prior* len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['ll_baseline_all_subjects_prior'] = ll_baseline_all_subjects_prior* len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['BICs_baseline_all_subjects_prior'] = BICs_baseline_all_subjects_prior* len(main_data_formatted['rule_name'][:n_rows])

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



          # post map trial specific data
          # main_data_formatted['raw_probs_post_map'] = raw_probs_post_map
          main_data_formatted['fitted_tau_post_map'] = fitted_taus_post_map
          main_data_formatted['log_ll_model_post_map'] = ll_model_post_map
          main_data_formatted['log_ll_baseline_post_map'] = ll_baseline_post_map
          main_data_formatted['BIC_model_post_map'] = BICs_model_post_map
          main_data_formatted['BIC_baseline_post_map'] = BICs_baseline_post_map
          # main_data_formatted['select_probs_fitted_tau_post_map'] = select_probs_post_map


          # post map data across trials
          # main_data_formatted['raw_probs_all_trials_post_map'] = raw_probs_all_trials_one_list_post_map
          main_data_formatted['fitted_tau_all_trials_post_map'] = fitted_taus_all_trials_post_map
          main_data_formatted['log_ll_model_all_trials_post_map'] = ll_model_all_trials_post_map
          main_data_formatted['log_ll_baseline_all_trials_post_map'] = ll_baseline_all_trials_post_map
          main_data_formatted['BIC_model_all_trials_post_map'] = BICs_model_all_trials_post_map
          main_data_formatted['BIC_baseline_all_trials_post_map'] = BICs_baseline_all_trials_post_map
          # main_data_formatted['select_probs_fitted_tau_all_trials_post_map'] = select_probs_post_map
          #
#           # print(fitted_taus_all_subjects_post_map)
#           # print(len(select_probs_all_subjects_post_map))
#           # print(ll_model_all_subjects_post_map)
#           # print(BICs_model_all_subjects_post_map)
#           # print(ll_baseline_all_subjects_prior)
#           # print(BICs_baseline_all_subjects_prior)
          main_data_formatted['fitted_taus_all_subjects_post_map'] = fitted_taus_all_subjects_post_map * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['raw_probs_all_subjects_post_map'] = [raw_probs_all_subjects_post_map] * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['ll_model_all_subjects_post_map'] = ll_model_all_subjects_post_map *len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['BICs_model_all_subjects_post_map'] = BICs_model_all_subjects_post_map* len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['ll_baseline_all_subjects_post_map'] = ll_baseline_all_subjects_prior * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['BICs_baseline_all_subjects_post_map'] = BICs_baseline_all_subjects_prior* len(main_data_formatted['rule_name'][:n_rows])

          # #
          # # post all specific data
            # post map trial specific data
          # main_data_formatted['raw_probs_post_all'] = raw_probs_post_all
          main_data_formatted['fitted_tau_post_all'] = fitted_taus_post_all
          main_data_formatted['log_ll_model_post_all'] = ll_model_post_all
          main_data_formatted['log_ll_baseline_post_all'] = ll_baseline_post_all
          main_data_formatted['BIC_model_post_all'] = BICs_model_post_all
          main_data_formatted['BIC_baseline_post_all'] = BICs_baseline_post_all
          # main_data_formatted['select_probs_fitted_tau_post_all'] = select_probs_post_all


          # post map data across trials
          # main_data_formatted['raw_probs_all_trials_post_all'] = raw_probs_all_trials_one_list_post_all
          main_data_formatted['fitted_tau_all_trials_post_all'] = fitted_taus_all_trials_post_all
          main_data_formatted['log_ll_model_all_trials_post_all'] = ll_model_all_trials_post_all
          main_data_formatted['log_ll_baseline_all_trials_post_all'] = ll_baseline_all_trials_post_all
          main_data_formatted['BIC_model_all_trials_post_all'] = BICs_model_all_trials_post_all
          main_data_formatted['BIC_baseline_all_trials_post_all'] = BICs_baseline_all_trials_post_all
          # main_data_formatted['select_probs_fitted_tau_all_trials_post_all'] = select_probs_all_trials_post_all
#           # print(fitted_taus_all_subjects_post_all)
#           # print(select_probs_all_subjects_post_all)
#           # print(ll_model_all_subjects_post_all)
#           # print(BICs_model_all_subjects_post_all)
#           # print(ll_baseline_all_subjects_prior)
#           # print(BICs_baseline_all_subjects_prior)
          main_data_formatted['fitted_taus_all_subjects_post_all'] = fitted_taus_all_subjects_post_all * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['raw_probs_all_subjects_post_all'] = [raw_probs_all_subjects_post_all] * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['ll_model_all_subjects_post_all'] = ll_model_all_subjects_post_all * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['BICs_model_all_subjects_post_all'] = BICs_model_all_subjects_post_all* len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['ll_baseline_all_subjects_post_all'] = ll_baseline_all_subjects_prior* len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['BICs_baseline_all_subjects_post_all'] = BICs_baseline_all_subjects_prior * len(main_data_formatted['rule_name'][:n_rows])


          # post all seed
          main_data_formatted['fitted_tau_post_all_seed'] = fitted_taus_post_all_seed
          main_data_formatted['log_ll_model_post_all_seed'] = ll_model_post_all_seed
          main_data_formatted['log_ll_baseline_post_all_seed'] = ll_baseline_post_map

          main_data_formatted['BIC_model_post_all_seed'] = BICs_model_post_all_seed
          main_data_formatted['BIC_baseline_post_all_seed'] = BICs_baseline_post_map
          # main_data_formatted['select_probs_fitted_tau_post_all'] = select_probs_post_all



          # main_data_formatted['raw_probs_all_trials_post_all'] = raw_probs_all_trials_one_list_post_all
          main_data_formatted['fitted_tau_all_trials_post_all_seed'] = fitted_taus_all_trials_post_all_seed
          main_data_formatted['log_ll_model_all_trials_post_all_seed'] = ll_model_all_trials_post_all_seed
          main_data_formatted['log_ll_baseline_all_trials_post_all_seed'] = ll_baseline_all_trials_post_all_seed
          main_data_formatted['BIC_model_all_trials_post_all_seed'] = BICs_model_all_trials_post_all_seed
          main_data_formatted['BIC_baseline_all_trials_post_all_seed'] = BICs_baseline_all_trials_post_all_seed
          # main_data_formatted['select_probs_fitted_tau_all_trials_post_all'] = select_probs_all_trials_post_all
#           # print(fitted_taus_all_subjects_post_all)
#           # print(select_probs_all_subjects_post_all)
#           # print(ll_model_all_subjects_post_all)
#           # print(BICs_model_all_subjects_post_all)
#           # print(ll_baseline_all_subjects_prior)
#           # print(BICs_baseline_all_subjects_prior)
          main_data_formatted['fitted_taus_all_subjects_post_all_seed'] = fitted_taus_all_subjects_post_all_seed * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['raw_probs_all_subjects_post_all_seed'] = [raw_probs_all_subjects_post_all_seed] * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['ll_model_all_subjects_post_all_seed'] = ll_model_all_subjects_post_all_seed * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['BICs_model_all_subjects_post_all_seed'] = BICs_model_all_subjects_post_all_seed * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['ll_baseline_all_subjects_post_all_seed'] = ll_baseline_all_subjects_prior * len(main_data_formatted['rule_name'][:n_rows])
          main_data_formatted['BICs_baseline_all_subjects_post_all_seed'] = BICs_baseline_all_subjects_prior * len(main_data_formatted['rule_name'][:n_rows])



          # labels comparing baseline performance with mcmc performance (1 = mcmc wins for this subject)
          main_data_formatted['prior_label'] = prior_labels
          main_data_formatted['prior_label_label'] = prior_labels_labels
          main_data_formatted['post_map_label'] = post_map_labels
          main_data_formatted['post_all_label'] = post_all_labels
          main_data_formatted['post_all_label_seed'] = post_all_labels_seed

          # accuracy of mcmc model predictions
          main_data_formatted['prior_acc'] = prior_accs
          main_data_formatted['prior_acc_single'] = prior_accs_single
          main_data_formatted['prior_label_acc'] = prior_label_accs
          main_data_formatted['prior_acc_single_label'] = prior_label_accs_single
          main_data_formatted['post_map_acc'] = post_map_accs
          main_data_formatted['post_map_acc_single'] = post_map_accs_single
          main_data_formatted['post_all_acc'] = post_all_accs
          main_data_formatted['post_all_acc_single'] = post_all_accs_single
          main_data_formatted['post_all_acc_seed'] = post_all_accs_seed
          main_data_formatted['post_all_acc_single_seed'] = post_all_accs_single_seed




          # maps
          main_data_formatted['map_prior'] = map_prior
          # main_data_formatted['map_prior_label'] = map_prior_label
          main_data_formatted['map_post_map'] = map_post_map
          main_data_formatted['map_post_all'] = map_post_all
          main_data_formatted['map_post_all_seed'] = map_post_all_seed

          # map accs
          main_data_formatted['map_prior_acc'] = map_prior_acc
          main_data_formatted['map_prior_acc_label'] = map_prior_acc_label
          main_data_formatted['map_post_map_acc'] = map_post_map_acc
          main_data_formatted['map_post_all_acc'] = map_post_all_acc
          main_data_formatted['map_post_all_acc_seed'] = map_post_all_acc_seed


          # all subjects average accuracy single tau
          main_data_formatted['all_sub_prior_acc'] = prior_acc_all_subj
          main_data_formatted['all_sub_prior_acc_label'] = prior_acc_all_subj_label
          main_data_formatted['all_sub_post_map_acc'] = post_map_acc_all_subj
          main_data_formatted['all_sub_post_all_acc'] = post_all_acc_all_subj
          main_data_formatted['all_sub_post_all_acc_seed'] = post_all_acc_all_subj_seed


          main_data_formatted['corr_rule_count_prior'] = correct_rule_perc_prior
          main_data_formatted['corr_rule_count_prior_label'] = correct_rule_perc_prior_labels
          main_data_formatted['corr_rule_count_prior_labels'] = correct_rule_perc_prior_labels
          main_data_formatted['correct_rule_perc_post_map'] = correct_rule_perc_post_map
          main_data_formatted['correct_rule_perc_post_all'] = correct_rule_perc_post_all
          main_data_formatted['correct_rule_perc_post_all_seed'] = correct_rule_perc_post_all_seed


          with open('model_results/test5_split/rules_prior_change_best_taut.txt', 'w') as filehandle:
              filehandle.writelines("%s\n" % place for place in rules_prior)
          with open('model_results/test5_split/rules_prior_label_change_best_taut.txt', 'w') as filehandle:
              filehandle.writelines("%s\n" % place for place in rules_prior_label)
          with open('model_results/test5_split/rules_post_map_change_best_taut.txt', 'w') as filehandle:
              filehandle.writelines("%s\n" % place for place in rules_post_map)
          with open('model_results/test5_split/rules_post_all_change_best_taut.txt', 'w') as filehandle:
              filehandle.writelines("%s\n" % place for place in rules_post_all)
          with open('model_results/test5_split/rules_post_all_seed_change_best_taut.txt', 'w') as filehandle:
              filehandle.writelines("%s\n" % place for place in rules_post_all_seed)





          main_data_formatted.to_csv('model_results/normative_res_two_' + str(rep) + '_process_models_change_best_taut.csv')   # writing main data to new csv file including all relevant data for analysis

          rep+=1

predicted_selections(main_data_formatted, rules_dict, replacements, trial_counts, n_rep=1)
