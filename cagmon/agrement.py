#---------------------------------------------------------------------------------------------------------#
import os
import sys
import csv
import math
import datetime
from os import makedirs
from os import listdir
import configparser

import numpy as np
import scipy as sp
from scipy.stats import pearsonr
from scipy.stats import kendalltau
import minepy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import multiprocessing
from multiprocessing import Process, Queue, cpu_count
queue = Queue()

import lalframe
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityFlag
from gwpy.time import tconvert
#---------------------------------------------------------------------------------------------------------#

###------------------------------------------### GENERAL ###-------------------------------------------###
# Make folders results saved
def Make_Directions(output_path, gst, get, main_channel, stride, sample_rate):
    if not output_path.split('/')[-1] == '':
        output_path = output_path + '/'
    html_folder_name = output_path + '{0}_{1}_{2}-{3}_{4}_{5}/'.format(tconvert(gst).strftime('%Y-%m-%d'), main_channel, int(gst), int(get), float(stride), int(sample_rate))
    folder_names = ['data', 'plots/Trend', 'plots/Scatter','plots/OmegaScan','segments']
    for folder_name in folder_names:
        path = html_folder_name + folder_name
        if not os.path.exists(path):
            os.makedirs(path)
    return html_folder_name
            
# Read Aux Channels from list
def Read_AuxChannels(main_channel, aux_channels_file_path):
    with open(aux_channels_file_path, 'r') as raw:
        AuxChannels = list()
        for row in raw.read().split('\n'):
            try :
                name = row.split(' ')[0]
                sample_rate = float(row.split(' ')[-1])
            except:
                name = row
                sample_rate = 'Unknown'
            if (name != main_channel) and (name != '') and (name.split(':')[0] == 'K1'):
                AuxChannels.append({'name':name, 'sample_rate':sample_rate})
    return AuxChannels

# Read Channel Lists from the frame file
def ChannelList_from_Frame(single_gwf):
    frfile = lalframe.FrameUFrFileOpen(single_gwf, 'r')
    frtoc = lalframe.FrameUFrTOCRead(frfile)
    nsim = lalframe.FrameUFrTOCQuerySimN(frtoc)
    for i in range(lalframe.FrameUFrTOCQueryAdcN(frtoc)):
        yield lalframe.FrameUFrTOCQueryAdcName(frtoc, i)
    for i in range(lalframe.FrameUFrTOCQueryProcN(frtoc)):
        yield lalframe.FrameUFrTOCQueryProcName(frtoc, i)
    for i in range(lalframe.FrameUFrTOCQuerySimN(frtoc)):
        yield lalframe.FrameUFrTOCQuerySimName(frtoc, i)

def Get_ChannelList(single_gwf):
    return sorted(ChannelList_from_Frame(single_gwf))

# Read configuration
def ReadConfig(ini_path):
    config = configparser.ConfigParser()
    if sys.version_info[0] >= 3:
        config.read(ini_path)
    else:
        config.read(unicode(ini_path,'utf-8'))
    
    # GENERAL
    gst = float(config['GENERAL']['gps_start_time'])
    get = float(config['GENERAL']['gps_end_time'])
    active_segment_only = config['GENERAL']['active_segment_only']
    show_additional_plots = config['GENERAL']['show_additional_plots']
    coefficients_trend_stride = float(config['GENERAL']['stride'])
    
    # PREPROSECCING
    try:
        datasize =  int(config['PREPROSECCING']['datasize'])
        sample_rate = datasize/coefficients_trend_stride
        if datasize == None:
            sample_rate = int(8192/coefficients_trend_stride)
        if not np.log2(sample_rate)%1 == 0.0:
            n = round(sample_rate)
            sample_rate = int(2**n)
    except KeyError:
        sample_rate = int(8192/coefficients_trend_stride)
        if not np.log2(sample_rate)%1 == 0.0:
            n = round(np.log2(sample_rate))
            sample_rate = int(2**n)
    try:
        whitening = config['PREPROSECCING']['whitening']
    except KeyError:
        whitening = 'no'
    try:
        rms =  config['PREPROSECCING']['rms']
    except KeyError:
        rms = 'no'
    try:
        filter_type =  config['PREPROSECCING']['filter_type']
        if not filter_type == None:
            if filter_type == 'lowpass' or filter_type == 'highpass':
                freq1 = float(config['PREPROSECCING']['frequency1'])
                freq2 = None
            elif filter_type == 'bandpass':
                freq1 = float(config['PREPROSECCING']['frequency1'])
                freq2 = float(config['PREPROSECCING']['frequency2'])
    except KeyError:
        filter_type = None
        freq1 = None
        freq2 = None
        
    # SEGMENT
    condition = str(config['SEGMENT']['defined_condition'])

    # CHANNELS
    main_channel = config['CHANNELS']['main_channel']
    aux_channels_file_path = config['CHANNELS']['aux_channels_file_path']

    # INPUT AND OUTPUT PATHS
    framefiles_path = config['INPUT AND OUTPUT PATHS']['frame_files_path']
    output_path = config['INPUT AND OUTPUT PATHS']['output_path']

    return gst, get, active_segment_only, show_additional_plots, coefficients_trend_stride, sample_rate, whitening, rms, filter_type, freq1, freq2, condition, main_channel, aux_channels_file_path, framefiles_path, output_path


###------------------------------------------### Read Timeseries data ###-------------------------------------------###
# Make Frame File List(ffl)
def GWF_Glue(framefiles_path, start, end):
    if not framefiles_path.split('/')[-1] == '':
        framefiles_path = framefiles_path + '/'
    gst = int(start)
    get = int(end)
    if (gst//100000) == (get//100000):
        cache = list()
        date = str(gst//100000)
        gwf_list = sorted(listdir(framefiles_path+date))
        truncated_gst = int((gst - (gst//100000)*100000)/32)*32 + (gst//100000)*100000
        truncated_get = math.ceil((get - (get//100000)*100000)/32)*32 + (get//100000)*100000     
        for i in range(len(gwf_list)):
            if truncated_gst <= int(gwf_list[i].split('-')[-2]) <= truncated_get:
                if gwf_list[i].split('.')[-1] == 'gwf' and gwf_list[i].split('.')[-1] == gwf_list[i].split('.')[1]:
                    cache.append(framefiles_path+date+'/'+gwf_list[i])
    else:
        cache = list()
        for n in range((get//100000) - (gst//100000) + 1):
            if n == 0:
                truncated_gst = int((gst - (gst//100000)*100000)/32)*32 + (gst//100000)*100000
                truncated_get = ((gst//100000)+1)*100000 - 32
                date = str(truncated_gst//100000)
                gwf_list = sorted(listdir(framefiles_path+date))
                for i in range(len(gwf_list)):
                    if truncated_gst <= int(gwf_list[i].split('-')[-2]) <= truncated_get:
                        if gwf_list[i].split('.')[-1] == 'gwf' and gwf_list[i].split('.')[-1] == gwf_list[i].split('.')[1]:
                            cache.append(framefiles_path+date+'/'+gwf_list[i])
            elif 0 < n < ((get//100000) - (gst//100000)):
                truncated_gst = ((gst//100000)+n)*100000
                truncated_get = ((gst//100000)+1+n)*100000 - 32
                date = str(truncated_gst//100000)
                gwf_list = sorted(listdir(framefiles_path+date))
                for i in range(len(gwf_list)):
                    if truncated_gst <= int(gwf_list[i].split('-')[-2]) <= truncated_get:
                        if gwf_list[i].split('.')[-1] == 'gwf' and gwf_list[i].split('.')[-1] == gwf_list[i].split('.')[1]:
                            cache.append(framefiles_path+date+'/'+gwf_list[i])              
            elif n == ((get//100000) - (gst//100000)):
                truncated_gst = ((gst//100000)+n)*100000
                truncated_get = math.ceil((get - (get//100000)*100000)/32)*32 + (get//100000)*100000
                date = str(truncated_gst//100000)
                gwf_list = sorted(listdir(framefiles_path+date))
                for i in range(len(gwf_list)):
                    if truncated_gst <= int(gwf_list[i].split('-')[-2]) <= truncated_get:
                        if gwf_list[i].split('.')[-1] == 'gwf' and gwf_list[i].split('.')[-1] == gwf_list[i].split('.')[1]:
                            cache.append(framefiles_path+date+'/'+gwf_list[i])                   
    cache = sorted(cache)
    return cache

# Organzeing pre-processing options
def PreprocessingOptions(whitening=None, rms=None, filter_type=None, freq1=None, freq2=None):
    options = dict()

    if not whitening == None:
        if whitening == 'yes' or whitening == 'no':
            options['whitening'] = whitening
        else:
            ValueError('Wrong whitening option was assigned, insert one of choice: yes or no')
    elif whitening == None:
        options['whitening'] = 'no'
        
    if not rms == None:
        if rms == 'yes' or rms == 'no':
            options['rms'] = rms
        else:
            ValueError('Wrong rms option was assigned, insert one of choice: yes or no')
    elif rms == None:
        options['rms'] = 'no'

    if not filter_type == None:
        if filter_type == 'lowpass':
            options['filter'] = ('lowpass', float(freq1))
        elif filter_type == 'highpass':
            options['filter'] = ('highpass', float(freq1))
        elif filter_type == 'bandpass':
            options['filter'] = ('bandpass', float(freq1), float(freq2))
        else:
            raise ValueError('Wrong filter name was assigned, insert one of name bellow: \n lowpass\n highpass\n bandpass')
    elif filter_type == None:
        options['filter'] = 'no'
        
    return options

# Read Timeseries data
def ReadTimeseries(cache, channel, gst, get, sample_rate=None, preprocessing_options=None, multiprocessing = False):
    try:
        data = TimeSeries.read(cache, channel, format='gwf.lalframe')
        data = data.crop(np.float64(gst), np.float64(get))
        if sample_rate == None and preprocessing_options == None:
            pass
        elif sample_rate != None and preprocessing_options == None:
            data = data.resample(float(sample_rate))

        elif sample_rate != None and preprocessing_options != None:
            whitening = preprocessing_options['whitening']
            rms = preprocessing_options['rms']
            filter_type = preprocessing_options['filter'][0]

            if whitening == 'yes':
                data = data.whiten()

            if not filter_type == 'no':
                if filter_type == 'lowpass':
                    freq = preprocessing_options['filter'][1]
                    data = data.lowpass(freq)
                elif filter_type == 'highpass':
                    freq = preprocessing_options['filter'][1]
                    data = data.highpass(freq)
                elif filter_type == 'bandpass':
                    freq1 = preprocessing_options['filter'][1]
                    freq2 = preprocessing_options['filter'][2]
                    if freq1 <= freq2:
                        data = data.bandpass(freq1, freq2)
                    else:
                        data = data.bandpass(freq2, freq1)
    
            if rms == 'yes':
                data = data.rms()

            if not float(sample_rate) == float(data.sample_rate.value):
                data = data.resample(float(sample_rate))
                
        if multiprocessing == True:
            queue.put(data)
    except RuntimeError:
        raise NameError('Wrong channel name was assigned: {}'.format(channel))
    return data   

# Parallel Process for Loading timeseries data and Coverting data type to numpy arrry
def Parallel_Load_data(cache, main_channel, AuxChannels, gst, get, sample_rate, preprocessing_options):
    number_of_cpus = cpu_count()
    All_channels = [main_channel]
    for row in AuxChannels:
        All_channels.append(row['name'])
    input_channel_list = list()
    if len(All_channels) <= number_of_cpus:
        input_channel_list.append(All_channels)
    elif len(All_channels) > number_of_cpus:
        for n in range(1+int(len(All_channels)/number_of_cpus)):
            if number_of_cpus*(n+1) < len(All_channels):
                input_channel_list.append(All_channels[number_of_cpus*n : number_of_cpus*(n+1)])
            elif number_of_cpus*(n+1) >= len(All_channels):
                input_channel_list.append(All_channels[number_of_cpus*(n) : ])
    data_list = list()
    for channel_segment in input_channel_list:
        procs = list()
        for channel in channel_segment:
            proc = Process(target=ReadTimeseries, args=(cache, channel, gst, get, sample_rate, preprocessing_options, True))
            procs.append(proc)
        for proc in procs:
            proc.start()
        for proc in procs:
            gotten_data = queue.get()
            data_list.append(gotten_data)
        for proc in procs:
            proc.join()       
    loaded_dataset = {'vanilla':dict(), 'array':dict()}
    empty_channels = list()
    for data in data_list:
        channel = data.name
        if np.size(data) == 0 or np.sum(np.isfinite(data)) ==  0:
            data = np.zeros(int((get-gst)*sample_rate))
            loaded_dataset['array'][channel] = np.array(data)
        elif np.size(data) != 0 and np.sum(np.isfinite(data)) !=  0:
            loaded_dataset['vanilla'][channel] = data
            loaded_dataset['array'][channel] = np.array(data)
        else:
            empty_channels.append(channel)
            print('{} is the Boolean channel and this channel will be ignored from remain process'.format(channel))
            pass
    print('Completely read {} of timeseries data except for empty channels'.format(len(All_channels) - len(empty_channels)))      
    return loaded_dataset


###------------------------------------------### Read and Make the segment file ###-------------------------------------------###
# Segment
def Segment(cache, gst, get, condition):
    if len(condition.split('>')) == 2:
        channel = condition.split('>')[0]
        if len(channel.split(' ')) >= 2:
            channel = channel.split(' ')[0]
        value = int(condition.split('>')[1])
        locked = ReadTimeseries(cache, channel, gst, get)
        locking = locked > value
    elif len(condition.split('>=')) == 2:
        channel = condition.split('>=')[0]
        if len(channel.split(' ')) >= 2:
            channel = channel.split(' ')[0]
        value = int(condition.split('>=')[1])
        locked = ReadTimeseries(cache, channel, gst, get)
        locking = locked >= value
    elif len(condition.split('==')) == 2:
        channel = condition.split('==')[0]
        if len(channel.split(' ')) >= 2:
            channel = channel.split(' ')[0]
        value = int(condition.split('==')[1])  
        locked = ReadTimeseries(cache, channel, gst, get)
        locking = locked == value
    elif len(condition.split('<')) == 2:
        channel = condition.split('<')[0]
        if len(channel.split(' ')) >= 2:
            channel = channel.split(' ')[0]
        value = int(condition.split('<')[1]) 
        locked = ReadTimeseries(cache, channel, gst, get)
        locking = locked < value
    elif len(condition.split('<=')) == 2:
        channel = condition.split('<=')[0]
        if len(channel.split(' ')) >= 2:
            channel = channel.split(' ')[0]
        value = int(condition.split('<=')[1])
        locked = ReadTimeseries(cache, channel, gst, get)
        locking = locked <= value
    segment = locking.to_dqflag(round=True)
    return segment

# Make segment XML and TXT with flag
def Segment_to_Files(output_path, segment, gst, get):
    if not output_path.split('/')[-1] == '':
        output_path = output_path + '/'
    try:
        segment.write('{0}segments/FlagSegment.xml'.format(output_path))
    except IOError:
        pass
    flag = segment.active
    flag_condition = segment.name
    flaged_segments = [flag_condition]
    if len(flag) == 1:
        if int(gst) == int(flag[0][0]) and int(get) == int(flag[0][1]):
            flaged_segments.append('{0} {1} {2}'.format(flag[0][0], flag[0][1], 'Active'))
        else:
            if int(gst) == flag[0][0]:
                flaged_segments.append('{0} {1} {2}'.format(flag[0][0],flag[0][1],'Active'))
                flaged_segments.append('{0} {1} {2}'.format(flag[0][1],int(get),'Inactive'))
            elif int(get) == flag[0][1]:
                flaged_segments.append('{0} {1} {2}'.format(int(gst),flag[0][0],'Inactive'))
                flaged_segments.append('{0} {1} {2}'.format(flag[0][0],flag[0][1],'Active'))
            else:
                flaged_segments.append('{0} {1} {2}'.format(int(gst),flag[0][0],'Inactive'))
                flaged_segments.append('{0} {1} {2}'.format(flag[0][0],flag[0][1],'Active'))
                flaged_segments.append('{0} {1} {2}'.format(flag[0][1],int(get),'Inactive'))
    elif len(flag) > 1:
        if int(gst) == int(flag[0][0]) and int(get) == int(flag[-1][1]):
            for i in range(len(flag)):
                flaged_segments.append('{0} {1} {2}'.format(flag[i][0], flag[i][1], 'Active'))
                if i < len(flag)-1:
                    flaged_segments.append('{0} {1} {2}'.format(flag[i][1], flag[i+1][0],'Inactive'))
        else:
            if int(gst) != int(flag[0][0]):
                flaged_segments.append('{0} {1} {2}'.format(int(gst), flag[0][0],'Inactive'))
            for i in range(len(flag)):
                flaged_segments.append('{0} {1} {2}'.format(flag[i][0], flag[i][1], 'Active'))
                if i < len(flag)-1:
                    flaged_segments.append('{0} {1} {2}'.format(flag[i][1], flag[i+1][0],'Inactive'))
            if int(get) != int(flag[-1][1]):
                flaged_segments.append('{0} {1} {2}'.format(flag[-1][1], int(get),'Inactive')) 
    f = open('{0}segments/FlagSegment.txt'.format(output_path), 'w')
    f.write('\n'.join(flaged_segments))         
    f.close() 

###------------------------------------------### Order and Pick values ###-------------------------------------------###
# Pick maximum value in given coefficient data
def Pick_maxvalues(output_path, gst, get, main_channel, stride, ctype):
    if not output_path.split('/')[-1] == '':
        output_path = output_path + '/'
    csv_path = output_path + 'data/{0}_trend_{1}-{2}_{3}-{4}.csv'.format(ctype, int(gst), int(get-gst), main_channel, int(stride)) 

    dict_bin = dict()
    with open(csv_path, 'r') as raw:
        for line in csv.DictReader(raw):
            tmp_bin = dict()
            channel = line['channel']
            values = list()
            for value in line.values():
                if not value == channel:
                    if value == 'nan':
                        values.append(0)
                    elif value != 'nan'and value != 'inactive':
                        values.append(float(value))
            median = np.median(values)
            if not median == 0: 
                for segment, value in line.items():
                    if segment != 'channel' and value != 'inactive':
                        if float(value) >= float(median):
                            tmp_bin[float(segment)] = float(value)
                max_value = max(tmp_bin.values())
                for gpstime, value in tmp_bin.items():
                    if value == float(max_value):
                        max_value_pair = [gpstime, value]
                        dict_bin[channel] = {'median':median, 'max_value': max_value_pair, 'values': tmp_bin}
    return dict_bin

# Sort maximum values by decrease odering 
def Sort_maxvalues(picked_maxvalues):
    maxinfo_bin = list()
    for channel in picked_maxvalues.keys():
        maxinfo_bin.append([channel, picked_maxvalues[channel]['max_value'][0], picked_maxvalues[channel]['max_value'][1]])
    sorted_maxinfo = sorted(maxinfo_bin, key=lambda item: item[2], reverse=True)

    return sorted_maxinfo
