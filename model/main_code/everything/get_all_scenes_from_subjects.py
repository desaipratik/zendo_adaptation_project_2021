import pandas as pd

subj_data = pd.read_csv('main_data_formatted_cond_one_second_rule.csv')


all_scenes = []
for scenes in subj_data['data_prior']:
    for scene in eval(scenes)[:8]:
        all_scenes.append(scene)




with open('1000_random_subj_scenes.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % place for place in all_scenes)

a = [1,2,3,4,5,6]
import random as rd
print(rd.sample(a,3))
