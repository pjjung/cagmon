
# -*- coding: utf-8 -*-
# Copyright (C) Phil Jung (2020)
#
# This file is port of CAGMon.
#
# CAGMon is the tool that evaluates the dependence between the primary and auxiliary channels of Gravitational-Wave detectors.
#
# CAGMon is following the GNU General Public License version 3. Under this term, you can redistribute and/or modify it.
# See the GNU free software license for more details.

'''main : main utility to read the GW channel data and the configuration file, and to conduct CAGMon algorithms.
'''

from cagmon.agrement import *
import cagmon.melody
import cagmon.conchord 
import cagmon.echo 

import argparse
import sys
parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("-v", "--version", action="store_true", help="Show version of CAGMon")
parser.add_argument("-c", "--config", action="store", type=str, help="the path of CAGMon configuration file")

args = parser.parse_args()
#---------------------------------------------------------------------------------------------------------#

if args.version:
    print('0.8.5')
    sys.exit()
if not args.config:
    parser.print_help()
    sys.exit()

__author__ = 'Phil Jung <pjjung@nims.re.kr>'

#---------------------------------------------------------------------------------------------------------#

def main():
    title='''
     ,-----.  ,---.   ,----.   ,--.   ,--.                            ,--.             ,--.        
    '  .--./ /  O  \ '  .-./   |   `.'   | ,---. ,--,--,      ,---. ,-'  '-.,--.,--. ,-|  | ,---.  
    |  |    |  .-.  ||  | .---.|  |'.'|  || .-. ||      \    | .-. :'-.  .-'|  ||  |' .-. || .-. : 
    '  '--'\|  | |  |'  '--'  ||  |   |  |' '-' '|  ||  |    \   --.  |  |  '  ''  '\ `-' |\   --. 
     `-----'`--' `--' `------' `--'   `--' `---' `--''--'     `----'  `--'   `----'  `---'  `----' 
    '''    
        
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
        
    number_of_cpus = cpu_count()
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    mem_gib = mem_bytes/(1024.**3)

    gst, get, coefficients_trend_stride, sample_rate, filter_type, freq1, freq2, condition, main_channel, aux_channels_file_path, framefiles_path, output_path = ReadConfig(args.config)

    if not framefiles_path.split('/')[-1] == '':
        framefiles_path = framefiles_path + '/'
    AuxChannels = Read_AuxChannels(main_channel, aux_channels_file_path)
    cache = GWF_Glue(framefiles_path, gst, get)    
        
    full_listdir = framefiles_path + str(int(gst/100000))
    for gwf in sorted(listdir(full_listdir)):
        if (int((gst - (gst//100000)*100000)/32)*32 + (gst//100000)*100000) == (int(gwf.split('-')[-2])):
            first_gwf = framefiles_path + str(int(gst/100000)) + '/' + gwf
    ch_ls = Get_ChannelList(first_gwf)
        
    print(HEADER+BOLD + title + ENDC)
    print(BOLD + '[Configuration Information]' + ENDC)
    print(' Start GPS time: {}'.format(gst))
    print(' End GPS time: {}'.format(get))
    print(' Main channels: {}'.format(main_channel))
    print(' Sample rate: {}Hz'.format(sample_rate))
    if filter_type == 'bandpass':
        print(' Bandpass filter option: {} ({}Hz - {}Hz)'.format(filter_type, freq1, freq2))
    elif filter_type == 'lowpass':
        print(' Lowpass filter option: {} ({}Hz)'.format(filter_type, freq1))
    elif filter_type == 'hingpass':
        print(' Highpass filter option: {} ({}Hz)'.format(filter_type, freq1))
    elif filter_type == None:
        print(' Band/Low/Highpass filter option: no')
    if condition[0] == 'condition':
        if len(condition[1].split('>=')) == 2:
            channel = condition[1].split('>=')[0]
            if len(channel.split(' ')) >= 2:
                channel = channel.split(' ')[0]
        elif len(condition[1].split('>')) == 2:
            channel = condition[1].split('>')[0]
            if len(channel.split(' ')) >= 2:
                channel = channel.split(' ')[0]
        elif len(condition[1].split('==')) == 2:
            channel = condition[1].split('==')[0]
            if len(channel.split(' ')) >= 2:
                channel = channel.split(' ')[0]
        elif len(condition[1].split('<=')) == 2:
            channel = condition[1].split('<=')[0]
            if len(channel.split(' ')) >= 2:
                channel = channel.split(' ')[0]
        elif len(condition[1].split('<')) == 2:
            channel = condition[1].split('<')[0]
            if len(channel.split(' ')) >= 2:
                channel = channel.split(' ')[0]
        print(' Defined segment condition: {}'.format(condition[1]))
    elif condition[0] == 'path':
        print(' Segment file path: {}'.format(condition[1]))
    print(' Coefficient trend stride: {} seconds'.format(coefficients_trend_stride))

    print(BOLD + '[Computing Resources]' + ENDC)
    print(' Given CPUs: {} cores'.format(number_of_cpus))
    print(' Given memory: {} GB'.format(mem_gib))
        
    print(BOLD + '[Configuration Validation Check]' + ENDC)
    OK = list()
    if len(cache)-2 <= (get-gst)/32 <= len(cache):
        print(OKBLUE + ' [OK] ' + ENDC + 'Cache')
        OK.append('OK')
    else:
        print(FAIL + ' [FAIL] ' + ENDC + 'Please check GPS times, it is not available to load the Time-series data within given duration')
        sys.exit()
    if main_channel in ch_ls:
        print(OKBLUE + ' [OK] ' + ENDC + 'Main channel')
        OK.append('OK')
    else:
        print(FAIL + ' [FAIL] ' + ENDC + 'Please check the name of the main channel or frame files, the main channel data did not exits in given frame files')
        sys.exit()
    aux_channel_check = list()
    for aux_channel in AuxChannels:
        if not aux_channel['name'] in ch_ls:
            aux_channel_check.append(aux_channel['name'])
    if len(aux_channel_check) == 0:
        print(OKBLUE + ' [OK] ' + ENDC + 'Aux-channels')
        OK.append('OK')
    else:
        print(FAIL + ' [FAIL] ' + ENDC + 'Please check the list of aux-channels, some aux-channels did not exist in given frame files')
        for aux_channel in aux_channel_check:
            print(FAIL + aux_channel + ENDC)
        sys.exit()

    if condition[0] == 'condition':
        if channel in ch_ls:
            print(OKBLUE + ' [OK] ' + ENDC + 'Segment')
            OK.append('OK')
        else:
            print(FAIL + ' [FAIL] ' + ENDC + 'Please check the segment condition, the given parameter did not exits in given frame files')
            sys.exit()
    elif condition[0] == 'path':
        if 'xml' == condition[1].split('.')[-1].lower() or 'json' ==  condition[1].split('.')[-1].lower():
            print(OKBLUE + ' [OK] ' + ENDC + 'Segment')
            OK.append('OK')
        else:
            print(FAIL + ' [FAIL] ' + ENDC + 'Please check the extension of the segment file, it must be xml or json')
            sys.exit()
    if sample_rate*coefficients_trend_stride > 1000:
        print(OKBLUE + ' [OK] ' + ENDC + 'Datasize')
        OK.append('OK')
    else:
        print(FAIL + ' [FAIL] ' + ENDC + 'Please check the stride, the given Datasize must has greater than 1 000')
        sys.exit()
    print(BOLD + '[Process Begins]' + ENDC)

    if len(OK) == 5:
        #Make folders
        output_path = Make_Directions(output_path, gst, get, main_channel, coefficients_trend_stride, sample_rate)
        
        #Calculate each coefficients
        print('Loading segments...')
        if condition[0] == 'condition':
            segment = Segment(cache, gst, get, condition[1])
            Segment_to_Files(output_path, segment, gst, get)
        elif condition[0] == 'path':
            segment = DataQualityFlag.read(condition[1])
            Segment_to_Files(output_path, segment, gst, get)
        preprocessing_options = PreprocessingOptions(filter_type, freq1, freq2)
        print('Calculating each coefficient...')
        cagmon.melody.Coefficients_Trend(output_path, framefiles_path, aux_channels_file_path, gst, get, coefficients_trend_stride, sample_rate, preprocessing_options, main_channel)

        MIC_maxvalues = Pick_maxvalues(output_path, gst, get, main_channel, coefficients_trend_stride, 'MICe')
        PCC_maxvalues = Pick_maxvalues(output_path, gst, get, main_channel, coefficients_trend_stride, 'PCC')
        Kendall_maxvalues = Pick_maxvalues(output_path, gst, get, main_channel, coefficients_trend_stride, 'Kendall')

        sorted_MIC_maxvalues = Sort_maxvalues(MIC_maxvalues)
        data_size = int(coefficients_trend_stride*sample_rate)
        mic_alpha, mic_c = cagmon.melody.MICe_parameters(data_size)

        AuxChannels = [{'name':x[0], 'sampling_rate':'Unknown'} for x in sorted_MIC_maxvalues]
        
        ## Make trend plots
        print('Plotting coefficient trend...')
        cagmon.conchord.Plot_Coefficients_Trend(output_path, gst, get, coefficients_trend_stride, main_channel, AuxChannels)
        
        ## Make coefficient distribution trend plots
        print('Plotting coefficient distribution trend...')
        for ctype in ['MICe', 'PCC', 'Kendall']:
            cagmon.conchord.Plot_Distribution_Trend(output_path, gst, get, main_channel, coefficients_trend_stride, ctype)

        #Make HTML file
        cagmon.echo.make_html(output_path, gst, get, coefficients_trend_stride, filter_type, freq1, freq2, main_channel, mic_alpha, mic_c, sample_rate, MIC_maxvalues, PCC_maxvalues, Kendall_maxvalues, sorted_MIC_maxvalues)
        
        print('\033[1m'+'\033[92m' + 'DONE' + '\033[0m')

if __name__ == '__main__':
    main()
