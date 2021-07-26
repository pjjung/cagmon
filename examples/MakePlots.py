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


def RetriveData(output_path, N, mic_alpha, mic_c):   
    NSRs = np.arange(0, 20, 0.5)
    
    power_data = dict()

    gaussian_power = np.empty((8, len(NSRs)))
    gamma_power = np.empty((8, len(NSRs)))
    brownian_power = np.empty((8, len(NSRs)))
    gw_power = np.empty((8, len(NSRs)))

    for i in range(len(NSRs)) :
        NSR = NSRs[i]
        csv_path = output_path+'power/power_{}_{}_a{}_c{}.csv'.format(N, NSR, mic_alpha, float(mic_c))
        with open(csv_path, 'r') as f:
            for line in csv.DictReader(f):
                j = int(line['function'])
                gaussian_power[j, i] = float(line['gaussian_power'])
                gamma_power[j, i] = float(line['gamma_power'])
                brownian_power[j, i] = float(line['brownian_power'])
                gw_power[j, i] = float(line['gw_power'])

    power_data = {'gaussian': gaussian_power, 
                 'gamma': gamma_power, 
                 'brownian': brownian_power, 
                 'gw': gw_power}
    
    return power_data

def PlotPower(output_path, mic_alpha, mic_c):
    data_sizes = [512, 1024, 2048, 4096, 8192]
    noise_types = ['gaussian', 'gw', 'gamma', 'brownian']
    for noise_type in noise_types:
        NSRs = np.arange(0, 20, 0.5)
        fig = plt.figure(1, figsize=(8, 20))
        #fig.suptitle(title, y=1.01 ,fontsize=18)
        for k, N in enumerate(data_sizes):
            plt.subplot(5,1,k+1)
            if noise_type == 'gaussian':
                title = 'Gaussian Noise (N: {} Alpha:{} c:{})'.format(N, mine_alpha, mine_c)
                loc='upper right'
                color_dict = {0:'black', 1:'darkred',2:'brown',3:'red',4:'coral',5:'orange',6:'darkorange',7:'grey',8:'darkgrey',9:'lightgrey',10:'white'}
            elif noise_type == 'gamma':
                title = 'Gamma Noise (N: {} Alpha:{} c:{})'.format(N, mine_alpha, mine_c)
                loc='lower right'
                color_dict = {0:'black', 1:'darkgreen',2:'green',3:'limegreen',4:'yellowgreen',5:'yellow',6:'gold',7:'grey',8:'darkgrey',9:'lightgrey',10:'white'}
            elif noise_type == 'brownian':
                title = 'Brownian Noise (N: {} Alpha:{} c:{})'.format(N, mine_alpha, mine_c)
                loc='lower right'
                color_dict = {0:'black', 1:'midnightblue',2:'blue',3:'royalblue',4:'skyblue',5:'aquamarine',6:'lightseagreen',7:'grey',8:'darkgrey',9:'lightgrey',10:'white'}
            elif noise_type == 'gw':
                loc='upper right'
                title = 'GW Detector Noise (N: {} Alpha:{} c:{})'.format(N, mine_alpha, mine_c)
                color_dict = {0:'black', 1:'navy',2:'blue',3:'deepskyblue',4:'violet',5:'magenta',6:'darkmagenta',7:'grey',8:'darkgrey',9:'lightgrey',10:'white'}

            plt.title(title)
            power_data, mu_null_data, sigma_null_data, mu_alt_data, sigma_alt_data = RetriveData(output_path, N, mine_alpha, mine_c)
            power = power_data[noise_type]

            function_names = ["Linear", "Quadratic", "Cubic", "Sine: period 1/2", "Sine: period 1/4", "Fourth root", "Circle", "Step function"]
            x_noise = NSRs
            for i in range(len(function_names)):
                plt.plot(x_noise, power[i], color=color_dict[i+1],label=function_names[i], marker=i, alpha=0.2 )
                if i == 0:
                    y = power[i]
                else:
                    y += power[i]

            plt.plot(x_noise, y/len(function_names), linewidth=2., label='Averaged',color=color_dict[1], marker='.', markersize=8, alpha=1 )    

            plt.ylabel("Power")
            plt.xlim((0, x_noise[-1]))
            plt.ylim((-0.05, 1.05))
            if k == 0 and (noise_type == 'gaussian' or noise_type == 'gw'):
                leg = plt.legend(loc=loc)
                leg_lines = leg.get_lines()
                plt.setp(leg_lines, linewidth=2)
        if not (noise_type == 'gaussian' or noise_type == 'gw'):
            leg = plt.legend(loc=loc)
            leg_lines = leg.get_lines()
            plt.setp(leg_lines, linewidth=2)   
        plt.xlabel("Noise-to-Signal Ratio(NSR)")
        plt.tight_layout()
        fig.savefig("{}summary/powers_{}_a{}_c{}.png".format(output_path,noise_type, mine_alpha, mine_c), bbox_inches='tight')
        plt.close()
        print('DONE')
        
def PlotHeatPowers(output_path):
    data_sizes = [512, 1024, 2048, 4096, 8192]
    noise_types = ['gaussian', 'gw', 'gamma', 'brownian']
    mic_alphas = np.arange(0.20,0.85,0.05)
    mic_cs = range(1,9)
    for noise_type in noise_types :
        fig = plt.figure(1, figsize=(8, 20))
        for k, N in enumerate(data_sizes):
            retrived_power = np.empty((len(mic_alphas), len(mic_cs)))
            for i in range(len(mic_alphas)):
                mic_alpha = mic_alphas[i]
                for j in range(len(mic_cs)):
                    mic_c = mic_cs[j]
                    try:
                        power_data, mu_null_data, sigma_null_data, mu_alt_data, sigma_alt_data = RetriveData(output_path, N, mic_alpha, mic_c)
                        powers = power_data[noise_type]
                    except:
                        powers = None
                    areas = list()
                    try:
                        for power in powers:
                            area = np.sum(power)*0.5
                            areas.append(area)
                        avg_area = np.mean(areas)
                    except:
                        avg_area = None
                    retrived_power[i,j] = avg_area
            ax = plt.subplot(5,1,k+1)
            x, y = np.meshgrid(mic_alphas, mic_cs)
            z = retrived_power.T 
            if noise_type == 'gaussian':
                title = 'Gaussian Noise (N: {})'.format(N)
                cmap = 'YlOrRd'
            elif noise_type == 'gamma':
                title = 'Gamma Noise (N: {})'.format(N)
                cmap = 'YlGn'
            elif noise_type == 'brownian':
                title = 'Brownian Noise (N: {})'.format(N)
                cmap = 'YlGnBu'
            elif noise_type == 'gw':
                title = 'GW Detector Noise (N: {})'.format(N)
                cmap = 'BuPu'
            ax.pcolormesh(x, y, z)
            ax.colorbar(cmap=cmap).set_label('Averaged AUPC') 
            ax.set_title(title, fontsize=18)
            ax.set_ylabel('c', fontsize=18)
        plt.xlabel('Alpha', fontsize=18)
        plt.subplots_adjust(wspace=10)
        plt.tight_layout()
        fig.savefig('{}summary/avgAUPC_{}.png'.format(output_path,noise_type), bbox_inches='tight')
        plt.close()
        print('Saved plot')

def PlotAveragedAUPC(output_path): ## old version
    mic_alphas = np.arange(0.20,0.85,0.05)
    mic_cs = range(1,9)
    data_sizes = [512, 1024, 2048, 4096, 8192]
    for N in data_sizes:
        fig = plt.figure(1, figsize=(8, 16))
        power_dict = {'gaussian':np.empty((len(mic_alphas), len(mic_cs))), 
                      'gw':np.empty((len(mic_alphas), len(mic_cs))),
                      'gamma':np.empty((len(mic_alphas), len(mic_cs))),
                     'brownian':np.empty((len(mic_alphas), len(mic_cs)))
                     }

        for i in range(len(mic_alphas)):
            mic_alpha = mic_alphas[i]
            for j in range(len(mic_cs)):
                mic_c = mic_cs[j]
                try:
                    power_data = RetriveData(output_path, N, mic_alpha, mic_c)
                except:
                    power_data = {'brownian':None, 'gaussian':None, 'gamma':None, 'gw':None}
                for noise_type in power_dict.keys():
                    powers = power_data[noise_type]
                    areas = list()
                    try:
                        for power in powers:
                            area = np.sum(power)*0.5
                            areas.append(area)
                        avg_area = np.mean(areas)
                    except:
                        avg_area = None
                    power_dict[noise_type][i,j] = avg_area

        for k, noise_type in enumerate(['gaussian', 'gw', 'gamma', 'brownian']):        
            ax = plt.subplot(4,1,k+1)

            x, y = np.meshgrid(mic_alphas, mic_cs)
            z = power_dict[noise_type].T 
            if noise_type == 'gaussian':
                title = 'Gaussian Noise (N: {})'.format(N)
            elif noise_type == 'gamma':
                title = 'Gamma Noise (N: {})'.format(N)
            elif noise_type == 'brownian':
                title = 'Brownian Noise (N: {})'.format(N)
            elif noise_type == 'gw':
                title = 'GW strain channel Noise (N: {})'.format(N)
            ax.pcolormesh(x, y, z)
            ax.colorbar(cmap='YlOrRd').set_label('Averaged AUPC')
            ax.set_title(title, fontsize=22)
            ax.set_ylabel('c', fontsize=18)
        plt.xlabel('Alpha', fontsize=18)
        plt.subplots_adjust(wspace=10)
        plt.tight_layout()
        fig.savefig('{}summary/avgAUPC_N{}.png'.format(output_path, N), bbox_inches='tight')
        plt.close()
        print('Saved plot')

def PlotCostNPower(output_path):
    noise_types = ['gaussian', 'gw', 'gamma', 'brownian']
    colors = ['dimgray', 'lightcoral', 'olive', 'darkorange', 'limegreen']
    markers = ['.', 'v', 'p', 'x', 'd']
    
    data_sizes = [512, 1024, 2048, 4096, 8192]
    mic_alphas = np.arange(0.20,0.85,0.05)
    mic_cs = range(1,8)
    normalized_factor = RunningTime('./', data_sizes[0], mic_alphas[0], mic_cs[0])
    
    fig = plt.figure(1, figsize=(8, 20))
    for k, noise_type in enumerate(noise_types):
        ax = plt.subplot(4,1,k+1)
        if noise_type == 'gaussian':
            title = 'Gaussian Noises'
        elif noise_type == 'gamma':
            title = 'Gamma Noises'
        elif noise_type == 'brownian':
            title = 'Brownian Noises'
        elif noise_type == 'gw':
            title = 'GW strain channel Noises'
        ax.set_title(title, fontsize=18)
        ax.set_ylabel('Relative computing cost (log scale)', fontsize=18)
        for N in data_sizes:
            most_powerfull = dict()
            for mic_alpha in mic_alphas:
                powerfull = dict()
                alpha_line_X = list()
                alpha_line_Y = list()
                for mic_c in mic_cs:
                    try:
                        power_data = RetriveData(output_path, N, mic_alpha, mic_c)
                    except:
                        power_data = {'brownian':None, 'gw':None, 'gaussian':None, 'gamma':None}
                    powers = power_data[noise_type]
                    areas = list()
                    try:
                        for power in powers:
                            area = np.sum(power)*0.5
                            areas.append(area)
                        avg_area = np.mean(areas)
                    except:
                        avg_area = None

                    avg_area = np.mean(areas)
                    cost = np.log10(RunningTime('./', N, round(mic_alpha,2), int(mic_c))/normalized_factor)

                    alpha_line_X.append(avg_area)
                    alpha_line_Y.append(cost)
                    if not np.isnan(avg_area) == True:
                        powerfull[avg_area] = cost
                    if mic_c%2 == 0:
                        ax.text(avg_area, cost, '  c{}'.format(mic_c), fontsize=10, color=colors[data_sizes.index(N)])
                    if mic_c == 1:
                        ax.text(avg_area, cost-0.1, '  a{}'.format(mic_alpha), fontsize=10, color=colors[data_sizes.index(N)])
                color = colors[data_sizes.index(N)]
                if mic_alpha == mic_alphas[-1]:
                    ax.plot(alpha_line_X, alpha_line_Y, color=color, marker=markers[data_sizes.index(N)], alpha=mic_alpha, label='N:{}'.format(N))
                else:
                    ax.plot(alpha_line_X, alpha_line_Y, color=color, marker=markers[data_sizes.index(N)], alpha=mic_alpha)
                try:
                    most_power = sorted(powerfull.keys(), reverse=True)[0]
                    most_powerfull[most_power] = powerfull[most_power]
                except:
                    pass
            marked_X = sorted(most_powerfull.keys(), reverse=True)[0]
            marked_Y = most_powerfull[marked_X]
            ax.plot(marked_X, marked_Y,markersize=10, color='r', marker='*')
    leg = plt.legend(fontsize=12, loc="lower right")
    for line in leg.get_lines():
        line.set_linewidth(2.0)
    plt.xlabel('Averaged AUPC', fontsize=18)
    plt.subplots_adjust(wspace=10)
    plt.tight_layout()
    fig.savefig("{}summar/powerNcost.png".format(output_path))
    plt.close()
    print('Saved plot')


def MostParameters(N, noise_type, percent):
    mic_alphas = np.arange(0.20,0.85,0.05)
    mic_cs = range(1,8)
    normalized_factor = RunningTime('./', data_sizes[0], mic_alphas[0], mic_cs[0])

    avgAUPCs = dict()
    for mic_alpha in mic_alphas:
        for mic_c in mic_cs:
            try:
                power_data = RetriveData(output_path, N, mic_alpha, mic_c)
            except:
                power_data = {'brownian':None, 'gw':None, 'gaussian':None, 'gamma':None}
            powers = power_data[noise_type]
            areas = list()
            try:
                for power in powers:
                    area = np.sum(power)*0.5
                    areas.append(area)
                avg_area = np.mean(areas)
            except:
                avg_area = None
            cost = RunningTime('./', N, round(mic_alpha,2), int(mic_c))/normalized_factor
            metadata={'alpha':round(mic_alpha,2), 'c':round(mic_c,2), 'cost':cost}
            avg_area = np.mean(areas)
            if not np.isnan(avg_area) == True:
                avgAUPCs[avg_area] = metadata
    print('NoiseType:{}, N: {}'.format(noise_type, N))
    print('alpha, c, averageAUPC, cost')
    for avgAUPC in sorted(avgAUPCs.keys(), reverse=True):
        if avgAUPC > sorted(avgAUPCs.keys(), reverse=True)[0]*np.float64(percent):
            alpha = avgAUPCs[avgAUPC]['alpha']
            c = avgAUPCs[avgAUPC]['c']
            cost = avgAUPCs[avgAUPC]['cost']
            print('{}, {}, {}, {}'.format(alpha, c, avgAUPC, cost))
