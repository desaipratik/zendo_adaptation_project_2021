"""
Created on Mon Jun 8 15:01:21 2020

@author: jan-philippfranken
"""
##############################################################################
# randomly creating scenes
###############################################################################

####################### General imports #######################################
import numpy as np
import random
from pruned_pcfg import pcfg_generator
ob = pcfg_generator()


def rand_scene_creator(rule, n_scenes=5,mean_n_objects = 2,sd_n_objects = 0,min_n_objects = 1,max_n_objects = 9):

    scene_list = []
    for scene in np.arange(n_scenes):
        n_objects = int(np.round(np.random.normal(mean_n_objects,sd_n_objects)))   # determine how many objects are in a particular scene
        while n_objects < min_n_objects or n_objects > max_n_objects:
            n_objects = int(np.round(np.random.normal(mean_n_objects,sd_n_objects)))
        objects_dict = {'ids': [],
                        'colours': [],
                        'sizes': [],
                        'xpos': [],
                        'ypos': [],
                        'rotations': [],
                        'orientations': [],
                        'contact': [],
                        'grounded': [],
                        'follow_rule': None}



        object_id = 0
        global X
        X = []
        obj_count = 0
        for object in np.arange(n_objects):      # determining the attributes of the specific objects in a scene
            colour = random.choice(['red','blue','green'])
            size = random.choice([1,2,3])
            xpos = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
            ypos = random.choice([2, 3, 4, 5, 6])
            rotation = random.choice(list(np.arange(0, 6.5, 0.5)))
            orientation = random.choice(['upright', 'lhs', 'rhs', 'strange'])
            contact = random.choice([[0],[0]])
            grounded = random.choice(['no', 'yes'])

            objects_dict['ids'].append(object_id)
            objects_dict['colours'].append(colour)
            objects_dict['sizes'].append(size)
            objects_dict['xpos'].append(xpos)
            objects_dict['ypos'].append(ypos)
            objects_dict['rotations'].append(rotation)
            objects_dict['orientations'].append(orientation)
            objects_dict['contact'].append(contact)
            objects_dict['grounded'].append(grounded)

            rand_object = {'id': object_id,
                           'colour': colour,
                           'size': size,
                           'xpos': xpos,
                           'ypos': ypos,
                           'rotation': rotation,
                           'orientation': orientation,
                           'contact': contact,
                           'grounded': grounded}

            X.append(rand_object)
            object_id += 1



        objects_dict['follow_rule'] = eval(rule)

        scene_list.append(objects_dict)
        # scene_list.append('##############################')

    # print(scene_list)
    return (scene_list)

# print(rand_scene_creator(rule="ob.exists(lambda x1: ob.exists(lambda x2: ob.and_operator(ob.and_operator(ob.and_operator(ob.and_operator(ob.and_operator(ob.equal(x1,'upright','orientation'),ob.equal(x1,'yes','grounded')),ob.equal(x2,'upright','orientation')),ob.equal(x2,'no','grounded')),ob.equal(x1,x2,'xpos')),ob.hor_operator(x1,x2,'contact')),X),X)", n_scenes=1))


# boolean rules
# there is a red = "ob.exists(lambda x1: ob.equal(x1, 'red','colour'), X)"
# nothing is upright = ob.forall(lambda x1: ob.not_operator(ob.equal(x1, 'upright', 'orientation')), X)
# one is blue = "ob.exactly(lambda x1: ob.equal(x1, 'blue','colour'), 1, X)"
# conjunct = "ob.exists(lambda x1: ob.and_operator(ob.equal(x1, 1,'size'),ob.equal(x1, 'blue','colour')), X)"
# disjunct = "ob.forall(lambda x1: ob.or_operator(ob.equal(x1, 1,'size'),ob.equal(x1, 'blue','colour')), X)"

# more complex rules
# match = ob.forall(lambda x1: ob.forall(lambda x2: ob.equal(x1,x2,'size'), X), X)"
# contact = "ob.exists(lambda x1: ob.exists(lambda x2: ob.hor_operator(x1,x2,'contact'), X), X)"
# specific relation = "ob.exists(lambda x1: ob.exists(lambda x2: ob.and_operator(ob.and_operator(ob.equal(x1, 'blue','colour'), ob.equal(x2 , 'red', 'colour')), ob.hor_operator(x1,x2,'contact')), X), X)"))"
# relative property = "ob.exists(lambda x1: ob.forall(lambda x2: ob.or_operator(ob.and_operator(ob.equal(x1,'red','colour'), ob.greater(x1,x2,'size')), ob.equal(x2, 'red', 'colour')), X), X)"
# stacked = "ob.exists(lambda x1: ob.exists(lambda x2: ob.and_operator(ob.and_operator(ob.and_operator(ob.and_operator(ob.and_operator(ob.equal(x1,'upright','orientation'),ob.equal(x1,'yes','grounded')),ob.equal(x2,'upright','orientation')),ob.equal(x2,'no','grounded')),ob.equal(x1,x2,'xpos'),ob.hor_operator(x1,x2,'contact')),X),X)"



# X = [{'id': 0, 'colour': 'red', 'size': 3, 'xpos': 5, 'ypos': 4, 'rotation': 3.1, 'orientation': 'upright', 'grounded': 'yes', 'contact': [0]},
#      {'id': 1, 'colour': 'red', 'size': 3, 'xpos': 2, 'ypos': 4, 'rotation': 5.0, 'orientation': 'rhs', 'grounded': 'yes', 'contact': [1]}]
#
# print(ob.exists(lambda x1: ob.forall(lambda x2: ob.and_operator(ob.and_operator(ob.equal(x1,'red','colour'), ob.not_operator(ob.equal(x2, 'red', 'colour'))), ob.greater(x1,x2,'size')), X), X))
# # print(ob.forall(lambda x1: ob.equal(x1, 'red', 'colour'), X))
# # print(ob.forall(lambda x1: ob.forall(lambda x2: ob.equal(x1,x2,'size'), X), X))
#
# #
# # ∃(λx1 : ∃(λx2 : ∧(∧(=
# # (x1 , blue, color), =
# # (x2 , red, color)), Γ(x1 , x2 , contact)), ), )
