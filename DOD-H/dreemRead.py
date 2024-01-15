import h5py
import os
import matplotlib.pyplot as plt
import numpy as np

# function to read the refrenced F3 signal, EOG and labels
def do(f1):
    sigs = f1['signals']
    eeg = sigs['eeg']    
    f3 = eeg['F3_M2']    
    eog = sigs['eog']    
    eog1 = eog['EOG1']    
    hyp = f1['hypnogram']
    return f3,eog1,hyp

# Function to extract single channel EEG data
def extract_data(ind, eeg_chan = 'F3_F4', path = ''):
    os.chdir(path)
    files = os.listdir()
    filess = []
    for i in files:
        if i.split('.')[-1]=='h5':
            filess.append(i)
    files = filess
    f1 = h5py.File(files[ind])
    signals = f1['signals']
    eeg = signals['eeg']
    x = eeg['FP1_F3']
    # x = eeg['F3_F4']
    # x = eeg['F3_M2']
    hyp = f1['hypnogram']
    x = np.array(x)
    return x, hyp

# function to extract the data from 2 different channels
def extract_data_bothChan(ind, eeg_chan = 'F3_F4', path = ''):
    os.chdir(path)
    files = os.listdir()
    filess = []
    for i in files:
        if i.split('.')[-1]=='h5':
            filess.append(i)
    files = filess
    f1 = h5py.File(files[ind])
    signals = f1['signals']
    eeg = signals['eeg']
    x = eeg['F3_F4']
    hyp = f1['hypnogram']
    x = np.array(x)
    x2 = eeg['F3_M2']
    hyp = f1['hypnogram']
    x2 = np.array(x2)
    return x, x2, hyp
