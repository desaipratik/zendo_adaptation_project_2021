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
from pcfg_generator import pcfg_generator
Z = pcfg_generator()


def rand_scene_creator(rule, n_scenes=5,mean_n_objects = 5,sd_n_objects = 1,min_n_objects = 3,max_n_objects = 9):

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
                        'grounded': []}
        object_id = 0
        global X
        X = []
        obj_count = 0
        for object in np.arange(n_objects):      # determining the attributes of the specific objects in a scene
            colour = random.choice(['blue','green','red'])
            size = random.choice([1,2,3])
            xpos = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
            ypos = random.choice([2, 3, 4, 5, 6])
            rotation = random.choice(list(np.arange(0, 6.5, 0.5)))
            orientation = random.choice(['upright','lhs','rhs','strange'])
            contact = random.choice([[[0, 2], [1], [2, 0]],[[0], [1], [2]]])
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

        # print(X)



        scene_list.append(objects_dict)
        # scene_list.append('##############################')

    # X = objects_dict
    # print(X)
    # objects_dict['follow_rule'] = eval(rule)

    # print(scene_list)
    return (scene_list)

# print(rand_scene_creator(mean_n_objects=3,rule="Z.exists(lambda x1: Z.equal(x1,'red','colour'),X)", n_scenes=1))
