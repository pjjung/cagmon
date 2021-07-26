import os
import sys
import csv
import math
from os import makedirs
from os import listdir
import random

import numpy as np
import scipy as sp
from scipy.signal import resample
import minepy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

##------------------------------------------------------------------------------------------##

import argparse
parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("-n", "--size", action="store", type=int)
parser.add_argument("-a", "--alpha", action="store", type=float)
parser.add_argument("-c", "--c", action="store", type=float)
parser.add_argument("-o", "--output", action="store", type=str)
args = parser.parse_args()

##------------------------------------------------------------------------------------------##

def Make_Directions(output_path):
    if not output_path.split('/')[-1] == '':
        output_path = output_path + '/'
    folder_names = ['power','dataforms','plots','summary']
    for folder_name in folder_names:
        path = output_path + folder_name
        if not os.path.exists(path):
            os.makedirs(path)
    return output_path

def ReadTimeseries(gwtxt):
    if gwtxt.split('.')[-1] == 'txt':
        if '16KHZ' in gwtxt.split('_')[-2]:
            samplerate = 16384
        elif '4KHZ' in gwtxt.split('_')[-2]:
            samplerate = 4096
        given_gpstime = int(gwtxt.split('-')[-2])
        duration = int(gwtxt.split('-')[-1].split('.')[0])

        dtype = np.dtype(np.float64, metadata={'sample_rate':samplerate, 't0':given_gpstime, 'duration':duration})
        data = np.loadtxt(gwtxt, dtype=dtype, comments='#')
        
        return data
    else:
        raise NameError('could not read this file format')

def CropTimeseries(data, gst, get):
    t0=data.dtype.metadata['t0']
    dur=data.dtype.metadata['duration']
    sr=data.dtype.metadata['sample_rate']

    cropped_data = data[(int(gst)-t0)*sr : (int(get)-t0)*sr]
    
    return cropped_data

def MICe(X, Y, alpha, c):
    mine = minepy.MINE(alpha=alpha, c=c, est="mic_e")
    mine.compute_score(X, Y)
    mic_value = mine.mic()
    return mic_value

def FunctionGen(N, x, nsr, fn, ntype): 
    functions = {"Linear" : x, 
                 "Quadratic" : 4 * (x-0.5)**2, 
                 "Cubic" : 128 * (x-1/3)**3 - 48 * (x-1/3)**2 - 12 * (x-1/3), 
                 "Sine: period 1/2" : np.sin(4*np.pi*x), 
                 "Sine: period 1/4" : np.sin(16*np.pi*x), 
                 "Fourth-root" : x**(0.25), 
                 "Circle" : (2*np.random.binomial(1, 0.5, N)-1) * (np.sqrt(1-(2*x-1)**2)), 
                 "Step function" : (x > 0.5)}
    ## Function Generation
    Signal = functions[fn] 
    if nsr == 0: ## noisless
        Y = Signal
    else:  ## with noises
        ## Noises
        if ntype == 'gaussian':
            Background = np.random.normal(0, 1, N)
        elif ntype == 'gamma':
            shape, scale = abs(np.random.randn(2)[0]), abs(np.random.randn(2)[1])
            Background = np.random.gamma(shape, scale, N) 
        elif ntype == 'brownian':
            x0 = np.random.randn(1)[0]
            Background = np.ones(N)*x0
            for i in range(1,N):
                # Sampling from the Normal distribution
                yi = np.random.normal()
                # Weiner process
                Background[i] = Background[i-1]+(yi/np.sqrt(N))
        elif  ntype == 'gwstrain': 
            while True:
                gw_event = random.choice(data_list)
                data = gw_event[0]
                start = gw_event[1]
                duration = gw_event[2]
                gst = np.random.choice(np.arange(start, start+duration), 1)[0]
                get = gst + 1
                sample_rate = N/1
                if sample_rate == 16384:
                    Background = CropTimeseries(data, gst, get)
                else:
                    Background = sp.signal.resample(CropTimeseries(data, gst, get), int(sample_rate))
                if np.sum(np.isnan(Background) == True) == 0:
                    break
        amp = np.mean(Background**2)**0.5/(np.mean(((nsr*Signal)**2)))**0.5
        Y = amp*Signal + Background           
    return Y

##------------------------------------------------------------------------------------------##

output_path = Make_Directions(args.output)
N = args.size
mine_alpha = args.alpha
mine_c = args.c

n_null = 500 #null datasets
n_alt = 500 #alternative datasets

##------------------------------------------------------------------------------------------##

## LIGO L1 strain data selection
gwf_list = list()
GWOpenData = '/data/kagra/home/piljong.jung/GWOpenData/txt/'
for event in sorted(listdir(GWOpenData)):
    for data_name in sorted(listdir(GWOpenData+event)):
        path = GWOpenData+event+'/'+data_name
        if data_name.split('-')[0] == 'L':
            gwf_list.append(path)

data_list = list()
for gwtxt in gwf_list:
    data = ReadTimeseries(gwtxt)
    start = data.dtype.metadata['t0']
    duration = data.dtype.metadata['duration']

    data_list.append([data, start, duration])
    
## Calculation of Statistical power 
function_names = ["Linear", "Quadratic", "Cubic", "Sine: period 1/2", "Sine: period 1/4", "Fourth-root", "Circle", "Step function"]
NSRs = np.arange(0, 20, 0.5)

gaussian_power = np.empty((len(function_names), len(NSRs)))
gamma_power = np.empty((len(function_names), len(NSRs)))
brownian_power = np.empty((len(function_names), len(NSRs)))
gw_power = np.empty((len(function_names), len(NSRs)))

np.random.seed(0) # Seed the generator

for i in range(len(NSRs)):
    NSR = NSRs[i]
    f_bin = [['function', 'N', 'NSR', 'Alpha', 'c',
              'gaussian_power', 'gamma_power', 'brownian_power', 'gw_power']]
    csv_path = output_path+'power/power_{}_{}_a{}_c{}.csv'.format(N, NSR, mine_alpha, mine_c)
    
    for j, function_name in enumerate(function_names):
        print("NSR: {}, function: {}".format(NSR, function_names[j]))

        gaussian_null, gamma_null, brownian_null, gw_null = [], [], [], []
        gaussian_alt, gamma_alt, brownian_alt, gw_alt    = [], [], [], []

        # null hypothesis
        for k in range(0, n_null):
            x = np.random.rand(N)
            y_gaussian = FunctionGen(N, x, NSR, function_name, 'gaussian')
            y_gamma = FunctionGen(N, x, NSR, function_name, 'gamma')
            y_brownian = FunctionGen(N, x, NSR, function_name, 'brownian')
            y_gw = FunctionGen(N, x, NSR, function_name, 'gwstrain')

            # resimulate x for the null scenario
            s = np.random.rand(N)

            gaussian_null.append( MICe(s, y_gaussian, mine_alpha, mine_c) )
            gamma_null.append( MICe(s, y_gamma, mine_alpha, mine_c) )
            brownian_null.append( MICe(s, y_brownian, mine_alpha, mine_c) )
            gw_null.append( MICe(s, y_gw, mine_alpha, mine_c) )
            
        # alternative hypothesis
        for k in range(0, n_alt):
            x = np.random.rand(N)
            y_gaussian = FunctionGen(N, x, NSR, function_name, 'gaussian')
            y_gamma = FunctionGen(N, x, NSR, function_name, 'gamma')
            y_brownian = FunctionGen(N, x, NSR, function_name, 'brownian')
            y_gw = FunctionGen(N, x, NSR, function_name, 'gwstrain')

            gaussian_alt.append( MICe(x, y_gaussian, mine_alpha, mine_c) )
            gamma_alt.append( MICe(x, y_gamma, mine_alpha, mine_c) )
            brownian_alt.append( MICe(x, y_brownian, mine_alpha, mine_c) )
            gw_alt.append( MICe(x, y_gw, mine_alpha, mine_c) )

        cut_gaussian = np.percentile(gaussian_null, 95) # statistical significance 5%
        cut_gamma = np.percentile(gamma_null, 95)
        cut_brownian = np.percentile(brownian_null, 95)
        cut_gw = np.percentile(gw_null, 95)

        gaussian_power[j, i] = np.sum(np.array(gaussian_alt) > cut_gaussian) / float(n_alt)
        gamma_power[j, i] = np.sum(np.array(gamma_alt) > cut_gamma) / float(n_alt)
        brownian_power[j, i] = np.sum(np.array(brownian_alt) > cut_brownian) / float(n_alt)
        gw_power[j, i] = np.sum(np.array(gw_alt) > cut_gw) / float(n_alt)
                
        print(' Gaussian noise {}'.format(gaussian_power[j, i]))    
        print(' Gamma noise {}'.format(gamma_power[j, i]))
        print(' Brownina noise {}'.format(brownian_power[j, i]) )
        print(' GW stain channel noise {}'.format(gw_power[j, i])) 
        
        f_bin.append([j ,N, NSR, mine_alpha, mine_c, 
                      gaussian_power[j, i], gamma_power[j, i], brownian_power[j, i], gw_power[j, i]])
        
    with open(csv_path, 'w') as f:
        csvwriter = csv.writer(f)
        for row in f_bin:
            csvwriter.writerow(row)
    f.close()  
    
fig = plt.figure(1, figsize=(14, 12))
x_noise = NSRs
for i in range(len(function_names)):
    plt.subplot(4,2,i+1)
    plt.title('{} (N:{} Alpha:{} c:{})'.format(function_names[i], N, mine_alpha, mine_c))
    plt.plot(x_noise, gaussian_power[i], label="Gaussian", color='orange', marker='.')
    plt.plot(x_noise, gw_power[i], label="GW", color='navy', marker='d')
    plt.plot(x_noise, gamma_power[i], label="Gamma", color='limegreen', marker='v')
    plt.plot(x_noise, brownian_power[i], label="Brownian", color='chocolate', marker='p')
        
    plt.xlabel("Noise-to-Signal Ratio(NSR)")
    plt.ylabel("Power")
    plt.xlim((0, x_noise[-1]))
    plt.ylim((-0.05, 1.05))
    leg = plt.legend(loc='upper right')
    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=2)
plt.tight_layout()
fig.savefig("{}plots/power_{}_a{}_c{}.png".format(output_path, N, mine_alpha, mine_c), bbox_inches='tight')
plt.close()
print('DONE')
