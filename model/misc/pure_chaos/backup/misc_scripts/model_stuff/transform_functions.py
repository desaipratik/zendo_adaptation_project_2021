import math

def compute_orientation(rot, epsilon = math.pi/6):
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
     contact_corrected = []
     for key in contact.keys():
          cone = key[1]
          if isinstance(contact[key], int):
               contact_corrected.append(int(cone) - 1)
          elif isinstance(contact[key], list):
               contact_corrected.append(contact[key])
     return contact_corrected

def check_structure(contact, i):
     if isinstance(contact[0], dict):
          key = list(contact[0].keys())[i]
          contact = compute_contact({key: contact[0][key]})
     elif isinstance(contact, list):
          contact = contact[i]
          if isinstance(contact, int):
               contact = [contact]
     return contact


def check_change_mult_feat(changes, truth):
    '''check if a change in two features of cones is associated with a change of the truth'''
    feats = ['a','b','c','d','e','f','g']
    ind = 0
    change_impact = {}
    for change in changes:
        change_impact[str(ind)] = {}
        subset = [x for i,x in enumerate(changes) if i!=ind]
        ind+=1
        ind_subset = 0
        for i in subset:
            ind_change =0
            ind_change_next = 1
            impact = []
            for i_2 in i:
                if ind_change_next == len(i):
                    ind_change_next = 0

                # impact of two features changing / or not changing on truth changing / or not changing
                # generic change (change - change in truth, no change, no change in truth)
                if change[ind_change] == True and truth[ind_change] == True and truth[ind_change_next] == False and i_2 == True:
                    impact.append(1)
                elif change[ind_change] == True and truth[ind_change] == False and truth[ind_change_next] == True and i_2 == True:
                    impact.append(1)

                elif change[ind_change] == False and truth[ind_change] == False and truth[ind_change_next] == False and i_2 == False:
                    impact.append(0)
                elif change[ind_change] == False and truth[ind_change] == True and truth[ind_change_next] == True and i_2 == False:
                    impact.append(1)

                # no change change in truth - penalize
                elif change[ind_change] == False and truth[ind_change] == True and truth[ind_change_next] == False and i_2 == False:
                    impact.append(-1)
                elif change[ind_change] == False and truth[ind_change] == False and truth[ind_change_next] == True and i_2 == False:
                    impact.append(-1)

                # change, no change in truth - penalize partially
                elif change[ind_change] == True and truth[ind_change] == True and truth[ind_change_next] == True and i_2 == True:
                    impact.append(-1)
                elif change[ind_change] == True and truth[ind_change] == False and truth[ind_change_next] == False and i_2 == True:
                    impact.append(0)

#                 # other feature change - truth change - reward
#                 elif change[ind_change] == False and truth[ind_change] == False and truth[ind_change_next] == True and i_2 == True:
#                     impact.append(1)
#                 elif change[ind_change] == False and truth[ind_change] == True and truth[ind_change_next] == False and i_2 == True:
#                     impact.append(1)

#                 # core feature change, truth change - reward
#                 elif change[ind_change] == True and truth[ind_change] == True and truth[ind_change_next] == False and i_2 == False:
#                     impact.append(1)
#                 elif change[ind_change] == True and truth[ind_change] == False and truth[ind_change_next] == True and i_2 == False:
#                     impact.append(1)



                # other feature change, no change in truth - penalize other

                elif change[ind_change] == False and truth[ind_change] == False and truth[ind_change_next] == False and i_2 == True:
                    impact.append(0)
                elif change[ind_change] == False and truth[ind_change] == True and truth[ind_change_next] == True and i_2 == True:
                    impact.append(-1)

                # other feature constant, no change in truth if core is change - penalize
                elif change[ind_change] == True and truth[ind_change] == True and truth[ind_change_next] == True and i_2 == False:
                    impact.append(-1)
                elif change[ind_change] == True and truth[ind_change] == False and truth[ind_change_next] == False and i_2 == False:
                    impact.append(0)





                ind_change+=1
                ind_change_next+=1
            change_impact[str(ind-1)][feats[ind_subset]] = sum(impact)
            ind_subset+=1
    return(change_impact)






def transform_xpos(xpositions):
    xposall = []
    for i in xpositions:
        xpos = []
        for i_2 in i:
            if i_2 <= 0.50:
                xpos.append(0)
            elif i_2 >= .50 and i_2 < 1.50:
                xpos.append(0)
            elif i_2 >= 1.50 and i_2 < 2.50:
                xpos.append(0)
            elif i_2 >= 2.50 and i_2 < 3.50:
                xpos.append(1)
            elif i_2 >= 3.50 and i_2 < 4.50:
                xpos.append(1)
            elif i_2 >= 4.50 and i_2 < 5.50:
                xpos.append(1)
            elif i_2 >= 5.50 and i_2 < 6.50:
                xpos.append(2)
            elif i_2 >= 6.50 and i_2 < 7.50:
                xpos.append(2)
            elif i_2 >= 7.50:
                xpos.append(2)
        xposall.append(xpos)
    return xposall

def transform_ypos(ypositions):
    yposall = []
    for i in ypositions:
        ypos = []
        for i_2 in i:
            if i_2 <= 2.50:
                ypos.append(2)
            elif i_2 >= 2.50 and i_2 < 3.50:
                ypos.append(3)
            elif i_2 >= 3.50 and i_2 < 4.50:
                ypos.append(4)
            elif i_2 >= 4.50:
                ypos.append(5)
        yposall.append(ypos)
    return yposall





def check_feat_ident(features):
    '''check if there is no change to a specific feature across all trial scenes
    resulting in exclusion of this feature given that the truth was stable as well'''
    equal = []
    for i in features[0]:
        for i_2 in features[1:]:
            for i_3 in i_2:
                if i_3 == i:
                    equal.append(1)
                else:
                    equal.append(0)
    print(equal)
    if len(equal) == sum(equal):
        return (True, equal)


def check_feat_change(features):
    '''check when features have changed from one trial scene to the next
    to establish covariation with True / False pattern'''
    ind_next = 0
    change_sum = []
    for i in features:
        change = []
        ind_next += 1
        if ind_next == len(features):
            ind_next = 0
        if set(i) == set(features[ind_next]):
            change.append(0)
        else:
            change.append(1)

#         for i_2 in i:
#             if set(i_2) =
#             for i_3 in features[ind_next]:
#                 if i_3 == i_2 or i == features[ind_next]:
#                     change.append(0)
#                 elif i_3 != i_2:
#                     change.append(1)
        if sum(change) == 0:
            change_sum.append(False)
        else:
            change_sum.append(True)
    return(change_sum)


def check_change_impact_single_feat(truth, change):
    '''check if a change in a feature of cones is associated with a change of the truth'''
    ind_next = 0
    ind_change = 0
    impact = []
    for i in truth:
        ind_next += 1
        if ind_next == len(truth):
            ind_next = 0
        # change in truth and change in feature
        if i == True and truth[ind_next] == False and change[ind_change] == True:
            impact.append(1)
        elif i == False and truth[ind_next] == True and change[ind_change] == True:
            impact.append(1)

        # no change in truth and no change in feature
        elif i == True and truth[ind_next] == True and change[ind_change] == False:
            impact.append(1)
        elif i == False and truth[ind_next] == False and change[ind_change] == False:
            impact.append(0)

        # change in truth and no change in feature
        elif i == False and truth[ind_next] == True and change[ind_change] == False:
            impact.append(-1)
        elif i == True and truth[ind_next] == False and change[ind_change] == False:
            impact.append(-1)

        # change in truth and change in feature
        elif i == True and truth[ind_next] == True and change[ind_change] == True:
            impact.append(-1)
        elif i == False and truth[ind_next] == False and change[ind_change] == True:
            impact.append(0)
        ind_change+=1
    return(impact)
