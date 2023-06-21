import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import math
import os
import pickle
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random


def Data_Loader(number_samples,all_key):
    data_lst = []
    main_directory = r"C:\Users\savva\OneDrive\Desktop\Data"
    for subdirectory in os.scandir(main_directory):
        for patients in os.scandir(subdirectory):
            patient_lst = []
            lock = 0
            try:
                for files in os.scandir(patients):
                    try: 
                        if str(pathlib.Path(subdirectory)).endswith('Negatives'):
                            label = 0
                        elif str(pathlib.Path(subdirectory)).endswith('Positives'):
                            label = 1
                        if str(pathlib.Path(files)).endswith('export.csv'):
                            dataload = pd.read_csv(files)
                            patient_lst.append(dataload)
                        elif str(pathlib.Path(files)).endswith('Chem_export.csv'):
                            dataload = pd.read_csv(files)
                            patient_lst.append(dataload)
                        if lock == 0:
                            patient_lst.append(label)
                            lock = 1
                    except: pass
            except: pass
            data_lst.append(patient_lst)
    data_array = []
    data_2D = []
    timer_2D = []
    labels = []
    if all_key == True:
        number_samples = len(data_lst) 
    for i in range(number_samples):
        try:
            data = data_lst[i][0] 
            time_vect = data['Time Elapsed']
            N0, M0, T0 = 56, 78, time_vect.shape[0]-1
            data2 = data_lst[i][2].iloc[:, :N0*M0]
            frame_2d = data2.to_numpy()
            frame_2d = clean_dead_pixels(frame_2d)
            frame_3d = np.zeros((M0, N0, T0))
            for a in range(T0):
                frame_3d[:, :, a] = frame_2d[a, :].reshape(M0, N0, order='F')  # take row and reshape. append to 3d array
            label = data_lst[i][1]
            data_array.append(frame_3d) 
            labels.append(label)
            data_2D.append(frame_2d)
            timer_2D.append(time_vect)

        except: print("Error - Sample:",i)
    return  timer_2D,data_2D,data_array, labels



def clean_dead_pixels(vector2D):
    time = np.size(vector2D,0)
    pixels = np.size(vector2D,1)
    # print("before",np.shape(vector2D))
    for i in range(pixels):
        change = 0
        for t in range(time-1):
           change = change + abs(vector2D[t][i]-vector2D[t+1][i])
        if change == 0: 
            for t in range(time):
                vector2D[t][i] < 1
    # print("after",np.shape(vector2D))
    return vector2D


timer,data2D,data3D,labels = Data_Loader(66,False)
