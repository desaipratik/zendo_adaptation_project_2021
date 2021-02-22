"""
Created on Mon Jun 8 13:24:21 2020

@author: jan-philippfranken
"""
###############################################################################
# This file contains functions used to transfrom data from the zendo task to
# enable more convenient evaluations of scenes and model runs
# It also includes functions for the adaptive PCFG case that return
# dictionaries scoring different feature values based on their relevance for a rule


####################### General imports #######################################
import math

def compute_orientation(rot, epsilon = math.pi/6):
    '''transform quantitative orientation feature into a categorical label'''
     orientations = []
     while rot > 2 * math.pi:
          rot = rot -2*math.pi
     while rot < 0:
          rot = rot+2*math.pi
     if abs(rot - math.pi) < epsilon:
          orientations.append('upright')
     elif abs(rot-1.2475) < epsilon:
           orientations.append('lhs')
     elif abs(rot-5.0375) < epsilon:
          orientations.append('rhs')
     else:
          orientations.append('strange')
     return orientations

def compute_contact(contact):
    '''check if two cones touch and return a list of lists instead of a dictionary which is returned by the raw
    javascript data'''
    contact_corrected = []
    for key in contact.keys():
        cone = key[1]
        if isinstance(contact[key], int):
            contact_corrected.append(int(cone) - 1)
        elif isinstance(contact[key], list):
            contact_corrected.append(contact[key])
    return contact_corrected

def check_structure(contact, i):
     '''again check if the structure in which contact is stored for a single scene
     is a list of lists and correct if necessary'''
     if isinstance(contact[0], dict):
          key = list(contact[0].keys())[i]
          contact = compute_contact({key: contact[0][key]})
     elif isinstance(contact, list):
          contact = contact[i]
          if isinstance(contact, int):
               contact = [contact]
     return contact


# def check_change_mult_feat(changes, truth):
#     '''check if a change in two features of cones is associated with a change of the truth'''
#     feats = ['a','b','c','d','e','f','g']
#     ind = 0
#     change_impact = {}
#     for change in changes:
#         change_impact[str(ind)] = {}
#         subset = [x for i,x in enumerate(changes) if i!=ind]
#         ind+=1
#         ind_subset = 0
#         for i in subset:
#             ind_change =0
#             ind_change_next = 1
#             impact = []
#             for i_2 in i:
#                 if ind_change_next == len(i):
#                     ind_change_next = 0
#
#                 # impact of two features changing / or not changing on truth changing / or not changing
#                 # generic change (change - change in truth, no change, no change in truth)
#                 if change[ind_change] == True and truth[ind_change] == True and truth[ind_change_next] == False and i_2 == True:
#                     impact.append(1)
#                 elif change[ind_change] == True and truth[ind_change] == False and truth[ind_change_next] == True and i_2 == True:
#                     impact.append(1)
#
#                 # elif change[ind_change] == False and truth[ind_change] == False and truth[ind_change_next] == False and i_2 == False:
#                 #     impact.append(0)
#                 # elif change[ind_change] == False and truth[ind_change] == True and truth[ind_change_next] == True and i_2 == False:
#                 #     impact.append(1)
#                 #
#                 # # no change change in truth - penalize
#                 # elif change[ind_change] == False and truth[ind_change] == True and truth[ind_change_next] == False and i_2 == False:
#                 #     impact.append(-1)
#                 # elif change[ind_change] == False and truth[ind_change] == False and truth[ind_change_next] == True and i_2 == False:
#                 #     impact.append(-1)
#
#                 # change, no change in truth - penalize partially
#                 elif change[ind_change] == True and truth[ind_change] == True and truth[ind_change_next] == True and i_2 == True:
#                     impact.append(-1)
#                 # elif change[ind_change] == True and truth[ind_change] == False and truth[ind_change_next] == False and i_2 == True:
#                 #     impact.append(0)
#
# #                 # other feature change - truth change - reward
# #                 elif change[ind_change] == False and truth[ind_change] == False and truth[ind_change_next] == True and i_2 == True:
# #                     impact.append(1)
# #                 elif change[ind_change] == False and truth[ind_change] == True and truth[ind_change_next] == False and i_2 == True:
# #                     impact.append(1)
#
# #                 # core feature change, truth change - reward
# #                 elif change[ind_change] == True and truth[ind_change] == True and truth[ind_change_next] == False and i_2 == False:
# #                     impact.append(1)
# #                 elif change[ind_change] == True and truth[ind_change] == False and truth[ind_change_next] == True and i_2 == False:
# #                     impact.append(1)
#
#
#
#                 # other feature change, no change in truth - penalize other
#                 #
#                 # elif change[ind_change] == False and truth[ind_change] == False and truth[ind_change_next] == False and i_2 == True:
#                 #     impact.append(0)
#                 # elif change[ind_change] == False and truth[ind_change] == True and truth[ind_change_next] == True and i_2 == True:
#                 #     impact.append(-1)
#                 #
#                 # # other feature constant, no change in truth if core is change - penalize
#                 # elif change[ind_change] == True and truth[ind_change] == True and truth[ind_change_next] == True and i_2 == False:
#                 #     impact.append(-1)
#                 # elif change[ind_change] == True and truth[ind_change] == False and truth[ind_change_next] == False and i_2 == False:
#                 #     impact.append(0)
#
#
#
#
#
#                 ind_change+=1
#                 ind_change_next+=1
#             change_impact[str(ind-1)][feats[ind_subset]] = sum(impact)
#             ind_subset+=1
#     return(change_impact)






# def transform_xpos(xpositions):
#     xposall = []
#     for i in xpositions:
#         xpos = []
#         for i_2 in i:
#             if i_2 <= 0.50:
#                 xpos.append(0)
#             elif i_2 >= .50 and i_2 < 1.50:
#                 xpos.append(1)
#             elif i_2 >= 1.50 and i_2 < 2.50:
#                 xpos.append(2)
#             elif i_2 >= 2.50 and i_2 < 3.50:
#                 xpos.append(3)
#             elif i_2 >= 3.50 and i_2 < 4.50:
#                 xpos.append(4)
#             elif i_2 >= 4.50 and i_2 < 5.50:
#                 xpos.append(5)
#             elif i_2 >= 5.50 and i_2 < 6.50:
#                 xpos.append(6)
#             elif i_2 >= 6.50 and i_2 < 7.50:
#                 xpos.append(7)
#             elif i_2 >= 7.50:
#                 xpos.append(8)
#         xposall.append(xpos)
#     return xposall
#
# def transform_ypos(ypositions):
#     yposall = []
#     for i in ypositions:
#         ypos = []
#         for i_2 in i:
#             if i_2 <= 2.50:
#                 ypos.append(2)
#             elif i_2 >= 2.50 and i_2 < 3.50:
#                 ypos.append(3)
#             elif i_2 >= 3.50 and i_2 < 4.50:
#                 ypos.append(4)
#             elif i_2 >= 4.50:
#                 ypos.append(5)
#         yposall.append(ypos)
#     return yposall
#
#
#

#
# def check_feat_ident(features):
#     '''check if there is no change to a specific feature across all trial scenes
#     resulting in exclusion of this feature given that the truth was stable as well'''
#     equal = []
#     for i in features[0]:
#         for i_2 in features[1:]:
#             for i_3 in i_2:
#                 if i_3 == i:
#                     equal.append(1)
#                 else:
#                     equal.append(0)
#     # print(equal)
#     if len(equal) == sum(equal):
#         return (True, equal)


def check_feat_change_both_directions(features,follow):
    '''check when features have changed from one trial scene to the next
    to establish covariation with True / False pattern and score those features and values that
    contribute to change, independent of the direction, which means that both presence and absence of features
    are scored'''
    val_set = set([val for sublist in features for val in sublist])
    val_score_dict = {el:0 for el in val_set}
    ind_next = 0
    change_sum = []
    for i in features:
        change = []
        ind_next += 1
        if ind_next == len(features):
            ind_next = 0
        if set(i) == set(features[ind_next]) and len(i) == len(features[ind_next]):
            change.append(0)
        else:
            change.append(1)
            set_init = set(i).difference(set(features[ind_next]))
            set_next = set(features[ind_next]).difference(set(i))
            if follow[features.index(i)] == True and follow[ind_next] == False:
                for value in set_init:
                    val_score_dict[value] += 1
                for value in set_next:
                    val_score_dict[value] += 1
            if follow[ind_next]== True and follow[features.index(i)]== False:
                for value in set_next:
                    val_score_dict[value] += 1
                for value in set_init:
                    val_score_dict[value] += 1
        if sum(change) == 0:
            change_sum.append(False)
        else:
            change_sum.append(True)

    return {'feat_change': change_sum, 'val_change': val_score_dict, 'sum_values': sum(val_score_dict.values())}



#
# def check_change_impact_single_feat(truth, change):
#     '''check if a change in a feature of cones is associated with a change of the truth'''
#     ind_next = 0
#     ind_change = 0
#     impact = []
#     for i in truth:
#         ind_next += 1
#         if ind_next == len(truth):
#             ind_next = 0
#         # change in truth and change in feature
#         if i == True and truth[ind_next] == False and change[ind_change] == True:
#             impact.append(1)
#         elif i == False and truth[ind_next] == True and change[ind_change] == True:
#             impact.append(1)
#
#         # # no change in truth and no change in feature
#         # elif i == True and truth[ind_next] == True and change[ind_change] == False:
#         #     impact.append(1)
#         # elif i == False and truth[ind_next] == False and change[ind_change] == False:
#         #     impact.append(0)
#         #
#         # # change in truth and no change in feature
#         # elif i == False and truth[ind_next] == True and change[ind_change] == False:
#         #     impact.append(-1)
#         # elif i == True and truth[ind_next] == False and change[ind_change] == False:
#         #     impact.append(-1)
#
#         # change in truth and change in feature
#         elif i == True and truth[ind_next] == True and change[ind_change] == True:
#             impact.append(-1)
#         # elif i == False and truth[ind_next] == False and change[ind_change] == True:
#         #     impact.append(0)
#         ind_change+=1
#     return(impact)
