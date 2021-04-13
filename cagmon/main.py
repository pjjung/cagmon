#!/usr/bin/env python2.7

#---------------------------------------------------------------------------------------------------------#
from cagmon.agrement import *
import cagmon.melody
import cagmon.conchord 
import cagmon.echo 

import argparse
import sys
parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("-v", "--version", action="store_true", help="Show version of CAGMon")
parser.add_argument("-c", "--config", action="store_true", type=str, help="the path of CAGMon configuration file")

args = parser.parse_args()
#---------------------------------------------------------------------------------------------------------#

if arge.version:
    print('0.8.0')
    sys.exit()
if not arge.config:
    parser.print_help()
    sys.exit()

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

    gst, get, active_segment_only, show_additional_plots, coefficients_trend_stride, sample_rate, whitening, rms, filter_type, freq1, freq2, condition, main_channel, aux_channels_file_path, framefiles_path, output_path = ReadConfig(args.config)

    if len(condition.split('>')) == 2:
        channel = condition.split('>')[0]
        if len(channel.split(' ')) >= 2:
            channel = channel.split(' ')[0]
    elif len(condition.split('>=')) == 2:
        channel = condition.split('>=')[0]
        if len(channel.split(' ')) >= 2:
            channel = channel.split(' ')[0]
    elif len(condition.split('==')) == 2:
        channel = condition.split('==')[0]
        if len(channel.split(' ')) >= 2:
            channel = channel.split(' ')[0]
    elif len(condition.split('<')) == 2:
        channel = condition.split('<')[0]
        if len(channel.split(' ')) >= 2:
            channel = channel.split(' ')[0]
    elif len(condition.split('<=')) == 2:
        channel = condition.split('<=')[0]
        if len(channel.split(' ')) >= 2:
            channel = channel.split(' ')[0]

    if not framefiles_path.split('/')[-1] == '':
        framefiles_path = framefiles_path + '/'
    AuxChannels = Read_AuxChannels(main_channel, aux_channels_file_path)
    cache = GWF_Glue(framefiles_path, gst, get)    
        
    full_listdir = framefiles_path + str(int(gst/100000))
    for gwf in sorted(listdir(full_listdir)):
        if (int((gst - (gst//100000)*100000)/32)*32 + (gst//100000)*100000) == (int(gwf.split('-')[-2])):
            first_gwf = framefiles_path + str(int(gst/100000)) + '/' + gwf   
    all_channels = os.popen('FrChannels {0}'.format(first_gwf)).read()
    all_channels_list = sorted(all_channels.split('\n'))
    ch_ls = list()
    for channels_list in all_channels_list :
        if channels_list.split(':')[0] == 'K1' :
            ch_ls.append(channels_list.split(' ')[0])
        
    print(HEADER+BOLD + title + ENDC)
    print(BOLD + '[Configuration Information]' + ENDC)
    print(' Start GPS time: {}'.format(gst))
    print(' End GPS time: {}'.format(get))
    print(' Main channels: {}'.format(main_channel))
    print(' Sample rate: {}Hz'.format(sample_rate))
    print(' Whitening option: {}'.format(whitening))
    print(' Active segment only option: {}'.format(active_segment_only))
    print(' Show additional plots option : {}'.format(show_additional_plots ))
    print(' Defined segment condition: {}'.format(condition))
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
    if channel in ch_ls:
        print(OKBLUE + ' [OK] ' + ENDC + 'Segment')
        OK.append('OK')
    else:
        print(FAIL + ' [FAIL] ' + ENDC + 'Please check the segment condition, the given parameter did not exits in given frame files')
        sys.exit()
    if sample_rate*coefficients_trend_stride > 1000:
        print(OKBLUE + ' [OK] ' + ENDC + 'Stride')
        OK.append('OK')
    else:
        print(FAIL + ' [FAIL] ' + ENDC + 'Please check the stride, the given stide must has greater than 1 000')
        sys.exit()
    print(BOLD + '[Process Begins]' + ENDC)

    if len(OK) == 5:
        #Make folders
        output_path = Make_Directions(output_path, gst, get, main_channel, coefficients_trend_stride, sample_rate)
        
        #Calculate each coefficients
        print('Loading segments...')
        segment = Segment(cache, gst, get, condition)

        Segment_to_Files(output_path, segment, gst, get)
        preprocessing_options = PreprocessingOptions(whitening, rms, filter_type, freq1, freq2)
        print('Calculating each coefficient...')
        if active_segment_only == 'no':
            melody.Coefficients_Trend(output_path, framefiles_path, aux_channels_file_path, gst, get, coefficients_trend_stride, sample_rate, preprocessing_options, main_channel)
        elif active_segment_only == 'yes':
            melody.Coefficients_Trend_Segment(output_path, framefiles_path, aux_channels_file_path, segment, gst, get, coefficients_trend_stride, sample_rate, preprocessing_options, main_channel)

        MIC_maxvalues = Pick_maxvalues(output_path, gst, get, main_channel, coefficients_trend_stride, 'MICe')
        PCC_maxvalues = Pick_maxvalues(output_path, gst, get, main_channel, coefficients_trend_stride, 'PCC')
        Kendall_maxvalues = Pick_maxvalues(output_path, gst, get, main_channel, coefficients_trend_stride, 'Kendall')

        sorted_MIC_maxvalues = Sort_maxvalues(MIC_maxvalues)
        data_size = int(coefficients_trend_stride*sample_rate)
        mic_alpha, mic_c = melody.MICe_parameters(data_size)

        AuxChannels = [{'name':x[0], 'sampling_rate':'Unknown'} for x in sorted_MIC_maxvalues]
        
        ## Make trend plots
        print('Plotting coefficient trend...')
        conchord.Plot_Coefficients_Trend(output_path, gst, get, coefficients_trend_stride, main_channel, AuxChannels)
        
        ## Make coefficient distribution trend plots
        print('Plotting coefficient distribution trend...')
        for ctype in ['MICe', 'PCC', 'Kendall']:
            if active_segment_only == 'no':
                conchord.Plot_Distribution_Trend(output_path, gst, get, main_channel, coefficients_trend_stride, ctype)
            elif active_segment_only == 'yes':
                conchord.Plot_Distribution_Trend_Segment(output_path, gst, get, main_channel, coefficients_trend_stride, ctype)

        ## Make Scatter and OmegaScan plots
        if show_additional_plots == 'yes':
            print('Plotting Scatter and OmegaScan plots')
            for max_MIC_info in sorted_max_MIC_list:
                aux_channel = max_MIC_info[0]
                marked_gst = max_MIC_info[1]
                marked_get = marked_gst + coefficients_trend_stride
                preprocessing_options = PreprocessingOptions(whitening, rms, filter_type, freq1, freq2)
                conchord.Scatter(framefiles_path, output_path, main_channel, aux_channel, gst, get, marked_gst, marked_get, sample_rate, preprocessing_options)
                if coefficients_trend_stride <= 30:
                    conchord.OmegaScan(framefiles_path, output_path, aux_channel, gst, get, marked_gst, marked_get, preprocessing_options)
                else:
                    conchord.Spectrogram(framefiles_path, output_path, aux_channel, gst, get, marked_gst, marked_get, preprocessing_options)
            
        #Make HTML file
        echo.make_html(output_path, gst, get, active_segment_only, show_additional_plots, coefficients_trend_stride, whitening, rms, filter_type, freq1, freq2, main_channel, mic_alpha, mic_c, sample_rate, MIC_maxvalues, PCC_maxvalues, Kendall_maxvalues, sorted_MIC_maxvalues)
        
        print('\033[1m'+'\033[92m' + 'DONE' + '\033[0m')

if __name__ == '__main__':
    main()
