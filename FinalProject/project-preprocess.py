# -*- coding: utf-8 -*-
"""
Martin Guy
Hannes Leskelä
December, 2016
Final Project KBLMM Course - UPC

Data : "Grasp-and-Lift EEG Detection" Kaggle dataset

This is a modified code of Alexandre Barachant (beat the benchmark), used to concentrate on the main part of the project
that was applying SVM to the data. It saves the pre-processed signals in .csv format.

"""

import numpy as np
import pandas as pd
from mne.io import RawArray
from mne.channels import read_montage
from mne.epochs import concatenate_epochs
from mne import create_info, find_events, Epochs, concatenate_raws, pick_types
from mne.decoding import CSP

from sklearn.linear_model import LogisticRegression
from glob import glob

from scipy.signal import butter, lfilter, convolve, boxcar
from joblib import Parallel, delayed


#### Defining useful functions for subsampling
## Many times redefined, depending on the one wanted.

#only take rows without 0
def take_good_subsample(data):
   
    labels = data[32:,:]
    
    
    (n,p) = np.shape(labels)
    
    filtre = []
    c = 0
    for i in range(p):
        if np.sum(labels[:,i]) > 0:
            c += 1
            filtre.append(True)
        else:
            filtre.append(False)
    
    filtre = np.array(filtre)
    
    fdata = data[:, filtre]
    fdata = fdata[:,::10]
    return fdata

#only take rows with ONLY one 1
def take_good_subsample(data):
    #fdata = data[:,::1000]
    #(n,p) = np.shape(data)
    
    labels = data[32:,:]
    
    #print("data:", n,p)
    
    (n,p) = np.shape(labels)
    #print("labels:", n,p)
    #print("On va compter, sur ", p, "lignes")
    
    filtre = []
    c = 0
    for i in range(p):
        if np.sum(labels[:,i]) == 1 :
            c += 1
            filtre.append(True)
        else:
            filtre.append(False)
    
    filtre = np.array(filtre)
    
    #print("Il y a", c, "lignes avec au moins un 1")
    #print("Maintenant on filtre")
    #labels = labels[:,filtre]
    
    #print("shape result:", np.shape(labels))
    
    #print("fini")
    fdata = data[:, filtre]
    fdata = fdata[:,::5]
    return fdata

	
#Now stuff related with MNE and signal preprocessing


def creat_mne_raw_object(fname,read_events=True, subsample=10):
    """Create a mne raw instance from csv file"""
    
    print("creat_mne_raw_object(%s)" % fname)

    # Read EEG file
    data = pd.read_csv(fname)
    #print(np.shape(data))
    # get chanel names
    ch_names = list(data.columns[1:])
    
    # read EEG standard montage from mne
    montage = read_montage('standard_1005',ch_names)

    ch_type = ['eeg']*len(ch_names)
    data = 1e-6*np.array(data[ch_names]).T
    
    if read_events:
        # events file
        ev_fname = fname.replace('_data','_events')
        # read event file
        events = pd.read_csv(ev_fname)
        events_names = events.columns[1:]
        events_data = np.array(events[events_names]).T
        
        # define channel type, the first is EEG, the last 6 are stimulations
        ch_type.extend(['stim']*6)
        ch_names.extend(events_names)
        # concatenate event file and data
        data = np.concatenate((data,events_data))
    
    data = take_good_subsample(data)
    #print(np.shape(data))
    
    # create and populate MNE info structure
    info = create_info(ch_names,sfreq=500.0, ch_types=ch_type, montage=montage)
    info['filename'] = fname
    
    # create raw object 
    raw = RawArray(data,info,verbose=False)
    
    return raw


subjects = range(1,4) #3 subjects

# design a butterworth bandpass filter 
freqs = [7, 30]
b,a = butter(5,np.array(freqs)/250.0,btype='bandpass')

# CSP parameters
# Number of spatial filter to use
nfilters = 4

# convolution
# window for smoothing features
nwin = 250

cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']



for subject in subjects:
	
	epochs_tot = []
	y = []
	
	print("> Read data")
	################ READ DATA ################################################
	fnames =  glob('train/subj%d_series*_data.csv' % (subject))
	
	raw = concatenate_raws([creat_mne_raw_object(fname) for fname in fnames])
	#raw, tdata = creat_mne_raw_object(fnames)
	print("All file read for this subject...")
	
	picks = pick_types(raw.info,eeg=True)
	raw._data[picks] = np.array(Parallel(n_jobs=-1)(delayed(lfilter)(b,a,raw._data[i]) for i in picks))
	
	
	
	print("> CSP Filters training")
	################ CSP Filters training #####################################
	# get event posision corresponding to HandStart
	print(">> Find events")
	events = find_events(raw,stim_channel='HandStart', verbose=False)
	# epochs signal for 2 second after the event
	print(">> Epoch signals")
	epochs = Epochs(raw, events, {'during' : 1}, 0, 2, proj=False,
					picks=picks, baseline=None, preload=True,
					add_eeg_ref=False, verbose=False)

	epochs_tot.append(epochs)
	y.extend([1]*len(epochs))

	# epochs signal for 2 second before the event, this correspond to the 
	# rest period.
	print(">> Epoch rest")
	epochs_rest = Epochs(raw, events, {'before' : 1}, -2, 0, proj=False,
					picks=picks, baseline=None, preload=True,
					add_eeg_ref=False, verbose=False)

	# Workaround to be able to concatenate epochs with MNE
	epochs_rest.times = epochs.times

	y.extend([-1]*len(epochs_rest))
	epochs_tot.append(epochs_rest)

	# Concatenate all epochs
	print(">> Concatenate epochs")
	epochs = concatenate_epochs(epochs_tot)

	# get data 
	print(">> get data")
	X = epochs.get_data()
	y = np.array(y)

	# train CSP
	print(">> Train CSP")
	csp = CSP(n_components=nfilters, reg='ledoit_wolf')
	csp.fit(X,y)
	
	################ Create Training Features #################################
	# apply csp filters and rectify signal
	print(">> apply CSP")
	feat = np.dot(csp.filters_[0:nfilters],raw._data[picks])**2

	# smoothing by convolution with a rectangle window    
	print(">> Smoothing")
	feattr = np.array(Parallel(n_jobs=-1)(delayed(convolve)(feat[i],boxcar(nwin),'full') for i in range(nfilters)))
	feattr = np.log(feattr[:,0:feat.shape[1]])

	# training labels
	# they are stored in the 6 last channels of the MNE raw object
	print(">> create labels")
	labels = raw._data[32:]
	
	
	print("Sauvegarde du signal train...")
	export_name = "train_filtered/filtered_"+str(subject)+'.csv'
	export_name_2 = "train_filtered/filtered_"+str(subject)+'_labels.csv'
	np.savetxt(export_name, np.transpose(feattr[:,:]), delimiter=",")
	np.savetxt(export_name_2, np.transpose(labels), delimiter=",")
	print("Done")