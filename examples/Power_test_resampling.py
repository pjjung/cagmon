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
        elif  ntype == 'gw': 
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

givenN1 = 8192
givenN2 = 1024
 
MIC_parameters = {'gaussian': {512:(0.4,1), 1024:(0.4,1), 2048:(0.45,2), 4096:(0.4,2), 8192:(0.4,2) },
                 'gw': {512:(0.4,7), 1024:(0.5,6), 2048:(0.5,6), 4096:(0.45,7), 8192:(0.45,7) },
                 'gamma': {512:(0.55,7), 1024:(0.5,6), 2048:(0.45,7), 4096:(0.4,7), 8192:(0.4,7) },
                 'brownian': {512:(0.6,6), 1024:(0.5,6), 2048:(0.5,7), 4096:(0.5,5), 8192:(0.5,7) }}

NSR = 2
n_null = 500
n_alt = 500

function_names = ["Linear", "Quadratic", "Cubic", "Sine: period 1/2", "Sine: period 1/4", "Fourth-root", "Circle", "Step function"]
scenario_names = {1:'HD', 2:'LU', 3:'BR'}
noise_types = ['gaussian', 'gw', 'gamma', 'brownian']

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
for noise_type in noise_types:
    fig = plt.figure(1, figsize=(8, 20))
    np.random.seed(0)
    
    for i, function_name in enumerate(function_names):
        ax1 = plt.subplot(8,1,i+1)
        ax2 = ax1.twinx()
        if noise_type == 'gaussian':
            title = '{} in Gaussian Noise (NSR: {})'.format(function_name, NSR)
            color = 'darkred'
        elif noise_type == 'gamma':
            title = '{} in Gamma Noise (NSR: {})'.format(function_name, NSR)
            color = 'darkgreen'
        elif noise_type == 'brownian':
            title = '{} in Brownian Noise (NSR: {})'.format(function_name, NSR)
            color = 'midnightblue'
        elif noise_type == 'gw':
            title = '{} in GW Detector Noise (NSR: {})'.format(function_name, NSR)
            color = 'navy'

        plt.title(title)

        densities_null = []
        densities_alt  = [] 
        powers = []

        f_bin = [['scenario', 'NSR', 'power']]
        csv_path = output_path+'power/resampling_power_{}_fn{}.csv'.format(noise_type, i)

        for j in scenario_names.keys():
            print("Scenario:{}, function: {}".format(j,  function_name))

            null = []
            alt  = []

            # null hypothesis
            for k in range(0, n_null):
                if j == 1: ## 8192 -> 1024
                    alpha = MIC_parameters[noise_type][givenN2][0]
                    c = MIC_parameters[noise_type][givenN2][1]

                    x_ = np.linspace(0, 1, givenN1, endpoint=False) # Make N:8192 datasets and Resample to N:1024
                    y = sp.signal.resample( FunctionGen(givenN1, x_, NSR, function_name, noise_type), givenN2)

                    # resimulate x for the null scenario
                    s = np.random.rand(givenN2)

                    null.append( MICe(s, y, alpha, c) )   

                elif j == 2: ## 1024 -> 8192 
                    alpha = MIC_parameters[noise_type][givenN1][0]
                    c = MIC_parameters[noise_type][givenN1][1]

                    x_ = np.linspace(0, 1, givenN2, endpoint=False) # Make N:1024 datasets and Resample to N:8192
                    y = sp.signal.resample( FunctionGen(givenN2, x_, NSR, function_name, noise_type), givenN1)
                    # resimulate x for the null scenario
                    s = np.random.rand(givenN1)

                    null.append( MICe(s, y, alpha, c) )

                elif j == 3: ## 8192, 1024 -> 2048 
                    targetN = 2048
                    alpha = MIC_parameters[noise_type][targetN][0]
                    c = MIC_parameters[noise_type][targetN ][1]

                    x_ = np.linspace(0, 1, givenN2, endpoint=False) # Make N:1024 datasets and Resample to N:2048
                    y = sp.signal.resample( FunctionGen(givenN2, x_, NSR, function_name, noise_type), targetN )
                    # resimulate x for the null scenario
                    s = np.random.rand(targetN )

                    null.append( MICe(s, y, alpha, c) )             

            # alternative hypothesis
            for k in range(0, n_alt):
                if j == 1: ## 8192 -> 1024
                    alpha = MIC_parameters[noise_type][givenN2][0]
                    c = MIC_parameters[noise_type][givenN2][1]

                    x_ = np.linspace(0, 1, givenN1, endpoint=False) # Make N:8192 datasets and Resample to N:1024
                    y = sp.signal.resample( FunctionGen(givenN1, x_, NSR, function_name, noise_type), givenN2)
                    x = np.linspace(0, 1, givenN2, endpoint=False) # Make N:1024 dataset

                    alt.append( MICe(x, y, alpha, c) )  

                elif j == 2: ## 1024 -> 8192 
                    alpha = MIC_parameters[noise_type][givenN1][0]
                    c = MIC_parameters[noise_type][givenN1][1]

                    x_ = np.linspace(0, 1, givenN2, endpoint=False) # Make N:1024 datasets and Resample to N:8192
                    y = sp.signal.resample( FunctionGen(givenN2, x_, NSR, function_name, noise_type), givenN1)
                    x = np.linspace(0, 1, givenN1, endpoint=False) # Make N:8192 dataset

                    alt.append( MICe(x, y, alpha, c) )

                elif j == 3: ## 8192, 1024 -> 2048
                    targetN = 2048
                    alpha = MIC_parameters[noise_type][targetN][0]
                    c = MIC_parameters[noise_type][targetN ][1]

                    x_ = np.linspace(0, 1, givenN2, endpoint=False) # Make N:1024 datasets and Resample to N:2048
                    y = sp.signal.resample( FunctionGen(givenN2, x_, NSR, function_name, noise_type), targetN )
                    x = sp.signal.resample( np.linspace(0, 1, givenN1, endpoint=False), targetN) # Make N:8192 datasets and Resample to N:2048

                    alt.append( MICe(x, y, alpha, c) )              

            cut = np.percentile(null, 95) # statistical significance 5%

            power = np.sum(np.array(alt) > cut) / float(n_alt)

            powers.append(power)
            densities_null.append(null) 
            densities_alt.append(alt)

            print(' {} noise, power: {}'.format(noise_type, power))    

            f_bin.append([j, NSR, power])

        vp_null = ax1.violinplot(densities_null, showmeans=False, showmedians=True)
        vp_alt = ax1.violinplot(densities_alt, showmeans=False, showmedians=True)
        power_scatter = ax2.scatter(scenario_names.keys(), powers, label='Power', color=color)
        #ax1.set_xlabel("Scenarios")     
        ax1.set_ylabel("Power")   
        ax2.set_ylabel('Strength of MICe', color='dimgray')
        ax2.set_yticks([])
        ax1.yaxis.grid(True)
        plt.yaxis.grid(True)
        ax1.set_ylim((-0.05, 1.05))
        ax2.set_ylim((-0.05, 1.05))
        plt.xticks(scenario_names.keys(), scenario_names.values())

        with open(csv_path, 'w') as f:
            csvwriter = csv.writer(f)
            for row in f_bin:
                csvwriter.writerow(row)
        f.close()  
    plt.legend([power_scatter, vp_null['bodies'][0],vp_alt['bodies'][0]],['Power','Null distribution', 'Alternative distribution'],
              bbox_to_anchor=(0., -0.35, 1., 0.102), loc='lower center', ncol=3, mode='expand', borderaxespad=0.)
    plt.tight_layout()
    fig.savefig("{}plots/resampling_powertest_{}.png".format(output_path, noise_type), bbox_inches='tight')
    plt.close()

    print('DONE')
