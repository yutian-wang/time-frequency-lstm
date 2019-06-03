#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 00:02:38 2018

@author: playfish
"""

import matplotlib  
#matplotlib.use('Agg')  
import matplotlib.pyplot as plt  
import numpy as np
import librosa
import librosa.display
from scipy import signal
import fnmatch
import os
import wave
import math
import pandas as pd
from keras.models import Sequential,Model
from keras.layers import Input,Dense,LSTM,TimeDistributed,Masking,Reshape,Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

os.environ['CUDA_VISIBLE_DEVICES']='0' 

audio_dir = './mir-1k/Wavfile/'
pv_dir = './mir-1k/PitchLabel/'
sample_rate = 16000
frame_size = 640
frame_shift = 320
fchunk_len = 64
fchunk_shift = 16
tchunk_len = 8
tchunk_shift = 1
freq_bins = int(sample_rate/frame_size)

    
def freq_to_midi(freq, ref_frequency=440.0):
    return 69.0 + 12.0*np.log2(freq/ref_frequency)

def midi_to_freq(midi_n, ref_frequency=440.0):
    if(midi_n==0):
        return 0
    else:
        return ref_frequency*2**((midi_n-69)/12)
    
def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def find_max_len(directory):
    files = find_files(directory)
    wavlens = []
    for filename in files:
        fn = wave.open(filename,'rb')
        wavlen = fn.getnframes()
        wavlens.append(wavlen)
        fn.close()
    return max(wavlens)

def load_generic_audio(directory, sample_rate):
    '''yields audio waveforms from the directory.'''
    files = find_files(directory)
    #files = files[0:50]
    for filename in files:
        audio, sr = librosa.load(filename, sr=sample_rate, mono=False)
        audio = audio[1]
        yield audio, filename

def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

def Discretizer(pitchs):
    pchs_df=pd.DataFrame(pitchs,columns=['frqs'])
    bins=np.arange(-freq_bins,sample_rate//2,freq_bins)
    dpchs=pd.cut(pchs_df.frqs,bins)
    dpchs=pd.DataFrame(dpchs)
    res=[]
    for index, row in dpchs.iterrows(): 
        res.append(row['frqs'].right)
    res=np.array(res)
    return res

def minmax_scale(x,x_min,x_max):
    return (x-x_min)/(x_max-x_min)
    
def inverse_minmax_scale(y,x_min,x_max):
    return x_min+(x_max-x_min)*y

def preprocess_audio(audio):
    if audio.size == 0:
        print("Warning: file size is 0!!!")
    audio = audio[frame_size:]
    spectra = librosa.stft(audio, n_fft = frame_size, hop_length = frame_shift)
    spectra_db = librosa.amplitude_to_db(np.abs(spectra), ref=np.max)
    data = spectra_db[:320,:].T
    time_len = data.shape[0]
    freq_len = data.shape[1]

    x = data[:, 0:fchunk_len]
    x = minmax_scale(x,-80,0)
    x = x.reshape(time_len, 1, fchunk_len)
    for i in range(1, (freq_len-fchunk_len)//fchunk_shift+1):
        t = data[:, i*fchunk_shift:(i*fchunk_shift+fchunk_len)]
        t = minmax_scale(t,-80,0)
        t = t.reshape(time_len, 1, fchunk_len)
        x = np.concatenate( (x,t), axis=1 )

    X_tmp=[]
    X_len=len(x)
    for i in range( (X_len-tchunk_len)//tchunk_shift+1 ):
        t = x[i*tchunk_shift:(i*tchunk_shift+tchunk_len),:,:]
        X_tmp.append(t)
    X_data = np.array(X_tmp)
    return X_data

def preprocess_label(filename):
    pitchs = []
    (filepath,tempfilename) = os.path.split(filename)
    (fname,extension) = os.path.splitext(tempfilename)
    pv_path = pv_dir + fname+'.pv'
    fn = open(pv_path,"r")
    lines = fn.readlines()  
    for line in lines:  
        pitchs.append( midi_to_freq(float(line) ) )
    pitchs = np.array(pitchs)
    
    pitchs = pitchs[tchunk_len-1:]    
    pitchs = Discretizer(pitchs)    
    return pitchs

def get_mir_data(train_pstg=0.8):

#    silence_threshold=0.3
    X_data = np.array([])
    Y_data = np.array([])

    iterator = load_generic_audio(audio_dir, sample_rate)
    idn=0
    for audio, filename in iterator:
        x = preprocess_audio(audio)

        if idn==0:
            X_data=x
        else:
            X_data=np.concatenate((X_data,x),axis=0)
        idn+=1
        

        pitchs = preprocess_label(filename)
        Y_data=np.concatenate((Y_data,pitchs),axis=0)
    
    enc = OneHotEncoder(categories='auto',sparse=False)
#    enc = OneHotEncoder(sparse=False)
    Y_data_onehot = enc.fit_transform(Y_data.reshape(-1,1))
#    Y_data_onehot = to_categorical(Y_data)
    

    split = int(len(X_data)*train_pstg)
    X_train = X_data[:split,:,:]
#    Y_train = Y_data[:split]
    Y_train_onehot = Y_data_onehot[:split,:]
    X_test = X_data[split:,:,:]
#    Y_test = Y_data[split:]
    Y_test_onehot = Y_data_onehot[split:,:]

    return (X_train, Y_train_onehot), (X_test, Y_test_onehot),enc

def inverse_X_data(data):
    
    time_chunk_num = data.shape[0]
    freq_chunk_num = data.shape[2]
    
    time_fuse=np.array([])
    for i in range(0,time_chunk_num,tchunk_len//tchunk_shift):
        t = data[i]
        if i==0:
            time_fuse = t
        else:           
            time_fuse = np.concatenate( (time_fuse,t), axis=0 )
            
    freq_fuse=np.array([])
    for i in range(0,freq_chunk_num,fchunk_len//fchunk_shift):
        t = time_fuse[:,i,:]
        if i==0:
            freq_fuse = t
        else:           
            freq_fuse = np.concatenate( (freq_fuse,t), axis=1 )      
            
    D = inverse_minmax_scale(freq_fuse,-80,0)
    return D.T

def inverse_Y_data(data,enc):
    Y_data = enc.inverse_transform(data)
    return Y_data
    
(X_train, Y_train_onehot), (X_test, Y_test_onehot), enc = get_mir_data()
time_batch=X_train.shape[1]
freq_batch=X_train.shape[2]
label_dim=Y_train_onehot.shape[1]


input_t = Input(shape=(time_batch, freq_batch, fchunk_len))

# Encodes a column of frequency bank using TimeDistributed Wrapper.
freq_lstm = TimeDistributed(LSTM(256,dropout=0.2))(input_t)

# Encodes row of encoded columns.
time_lstm = LSTM(256, return_sequences = True,dropout=0.2)(freq_lstm)
time_lstm = LSTM(256, return_sequences = True,dropout=0.2)(time_lstm)
time_lstm = LSTM(256, return_sequences = False,dropout=0.2)(time_lstm)

# Final predictions and model.
prediction = Dense(label_dim, activation='softmax')(time_lstm)

model = Model(input_t, prediction)
model.summary()

opt=RMSprop(lr=0.0005)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(X_train, Y_train_onehot, epochs=50, batch_size=64,validation_data=(X_test, Y_test_onehot))

model.save('pv_mlstm.h5')
test=X_test[0:600]
result = model.predict(test)
D=inverse_X_data(test)
test_Y=Y_test_onehot[0:600]
Y_labels=inverse_Y_data(test_Y,enc)
Y_preds=inverse_Y_data(result,enc)

plt.figure(figsize=(D.shape[1]//30,D.shape[0]//30))
librosa.display.specshow(D, y_axis='log',x_axis='time',sr=sample_rate,hop_length=frame_shift)
plt.colorbar(format='%+2.0f dB')
plt.title('spectrogram')
x_coords = librosa.display.__coord_time(D.shape[1],sr=sample_rate,hop_length=frame_shift)
plt.plot(x_coords[:x_coords.size-1],Y_labels,'g-')
plt.plot(x_coords[:x_coords.size-1],Y_preds,'r.')
plt.savefig('res5.png')
