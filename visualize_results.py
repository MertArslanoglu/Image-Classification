import os
import pickle
from utils import part3Plots, part4Plots, part5Plots, visualizeWeights

path = "./part_3_results"
#  dictionaries saved at the path
dir_list = os.listdir(path)
print(dir_list)
dict_list = []
#  open dictionaries and append to list
for dir in dir_list :
    with open('./part_3_results/{}'.format(dir), 'rb') as handle:
        dict_list.append(pickle.load(handle))
part3Plots(dict_list, save_dir='./graphs', filename='part_3_results', show_plot=True)

path = "./part_4_results"
#  dictionaries saved at the path
dir_list = os.listdir(path)
print(dir_list)
dict_list = []
#  open dictionaries and append to list
for dir in dir_list :
    with open('./part_4_results/{}'.format(dir), 'rb') as handle:
        dict_list.append(pickle.load(handle))
part4Plots(dict_list, save_dir='./graphs', filename='part_4_results', show_plot=True)


path = "./part_5_results"
#  dictionaries saved at the path
dir_list = os.listdir(path)
print(dir_list)
dict_list = []
#  open dictionaries and append to list
for dir in dir_list :
    with open('./part_5_results/{}'.format(dir), 'rb') as handle:
        dict_list.append(pickle.load(handle))
part3Plots(dict_list, save_dir='./graphs', filename='part_5_results', show_plot=True)

