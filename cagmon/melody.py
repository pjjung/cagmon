
# -*- coding: utf-8 -*-
# Copyright (C) Phil Jung (2020)
#
# This file is port of CAGMon.
#
# CAGMon is the tool that evaluates the dependence between the primary and auxiliary channels of Gravitational-Wave detectors.
#
# CAGMon is following the GNU General Public License version 3. Under this term, you can redistribute and/or modify it.
# See the GNU free software license for more details.

'''melody : the core utility for calculating each coefficient.
'''

from cagmon.agrement import *

__author__ = 'Phil Jung <pjjung@nims.re.kr>'

###------------------------------------------### Coefficients ###-------------------------------------------###
# PCC
def PCC(loaded_dataset, main_channel):
    result_bin = dict()
    aux_channels = [channel for channel in loaded_dataset['array'].keys()]
    aux_channels.remove(main_channel)
    hoft_data = loaded_dataset['array'][main_channel]
    
    for aux_channel in aux_channels:
        aux_data = loaded_dataset['array'][aux_channel]
        if ((hoft_data**2).sum())**(0.5) == 0.0 or ((aux_data**2).sum())**(0.5) == 0.0:
            R = 0.
            print('PCC\n Channel: {0}\n Value: {1}'.format(aux_channel, R))
        else:
            hoft_data = hoft_data/((hoft_data**2).sum())**(0.5)
            aux_data = aux_data/((aux_data**2).sum())**(0.5)
        
            mx = aux_data.mean()
            my = hoft_data.mean()
            xm, ym = aux_data-mx, hoft_data-my
            if sum(xm)*sum(ym) != 0:
                R = abs(pearsonr(aux_data, hoft_data)[0])
            elif sum(xm)*sum(ym) == 0 or np.isfinite(R) == True:
                R = 0.
            result_bin[aux_channel] = R
            print('PCC\n Channel: {0}\n Value: {1}'.format(aux_channel, R))

    return result_bin

# Kendall's tau
def Kendall(loaded_dataset, main_channel):
    result_bin = dict()
    aux_channels = [channel for channel in loaded_dataset['array'].keys()]
    aux_channels.remove(main_channel)
    hoft_data = loaded_dataset['array'][main_channel] 
    for aux_channel in aux_channels:
        aux_data = loaded_dataset['array'][aux_channel]
        tau = abs(kendalltau(aux_data, hoft_data)[0])
        if not np.isfinite(tau) == True:
            tau = 0.
        result_bin[aux_channel] = tau
        print('Kendall\n Channel: {0}\n Value: {1}'.format(aux_channel, tau))

    return result_bin

# Estimate appropriate value of Alpha and c for MICe
def MICe_parameters(data_size):
    NPOINTS_BINS = [1,    25,   50,   250,   500, 1000, 2500, 4000, 8000, 10000, 40000]
    ALPHAS =       [0.85, 0.80, 0.75, 0.70, 0.55, 0.5,  0.55, 0.55, 0.5,  0.45,  0.4]
    CS =           [5,    5,    5,    5,     7,   7,    6,    6,    0.7,  1,     1]
    if data_size < 1:
        raise ValueError("the number of data size must be >=1")

    alpha = ALPHAS[np.digitize([data_size], NPOINTS_BINS)[0] - 1]
    c = CS[np.digitize([data_size], NPOINTS_BINS)[0] - 1]
    return alpha, c

# MICe for multiprocessinf queue
def Queue_MIC(loaded_dataset, main_channel, aux_channel):
    result_bin = list()
    hoft_data = loaded_dataset['array'][main_channel]
    aux_data = loaded_dataset['array'][aux_channel]
    data_size = int(hoft_data.size)
    alpha, c = MICe_parameters(data_size)
    mine = minepy.MINE(alpha=alpha, c=c, est="mic_e")
    mine.compute_score(aux_data, hoft_data)
    mic_value = mine.mic()
    print('MICe\n Channel: {0}\n Value: {1}'.format(aux_channel, mic_value))
    result_bin.append([aux_channel, mic_value])
    queue.put(result_bin)  

# Calculate MICe parallely
def Parallel_MIC(loaded_dataset, main_channel):
    number_of_cpus = cpu_count()
    aux_channels = [channel for channel in loaded_dataset['array'].keys()]
    aux_channels.remove(main_channel)
    input_channel_list = list()
    if len(aux_channels) <= number_of_cpus:
        input_channel_list.append(aux_channels)
    else:
        for n in range(1+int(len(aux_channels)/number_of_cpus)):
            if number_of_cpus*(n+1) < len(aux_channels):
                input_channel_list.append(aux_channels[number_of_cpus*n : number_of_cpus*(n+1)])
            elif number_of_cpus*(n+1) >= len(aux_channels):
                input_channel_list.append(aux_channels[number_of_cpus*(n) : ])
    data_list = list()   
    for channel_segment in input_channel_list:
        procs = list()
        for channel in channel_segment:
            proc = Process(target=Queue_MIC, args=(loaded_dataset, main_channel, channel))
            procs.append(proc)
        for proc in procs:
            proc.start()
        for proc in procs:
            gotten_data = queue.get()
            data_list.extend(gotten_data)
        for proc in procs:
            proc.join()    
    result_bin = dict()
    for row in data_list:
        result_bin[row[0]] = row[1]
    return result_bin

###------------------------------------------### Trend ###-------------------------------------------###
# Coefficient trend
def Coefficients_Trend(output_path, framefiles_path, aux_channels_file_path, gst, get, stride, sample_rate, preprocessing_options, main_channel):
    if not output_path.split('/')[-1] == '':
        output_path = output_path + '/'
    dicts_bin = dict()
    if sample_rate * stride < 1000: 
        raise ValueError('These arguments are not vailable if the number of valiables in a set is less than 1 000')
    else:
        segments = np.arange(gst, get, stride)
        for segment in segments:
            start = segment
            end = start + stride
            print('# Segment[{0}/{1}]: {2} - {3} (stride: {4} seconds)'.format(1+list(segments).index(segment), len(segments), start, end, stride))
            cache = GWF_Glue(framefiles_path, start, end)
            AuxChannels = Read_AuxChannels(main_channel, aux_channels_file_path)   
            loaded_dataset = Parallel_Load_data(cache, main_channel, AuxChannels, start, end, sample_rate, preprocessing_options)
            print('Calculating PCC coefficients...')
            PCC_dict = PCC(loaded_dataset, main_channel)
            print('Calculating Kendall coefficients...')
            Kendall_dict = Kendall(loaded_dataset, main_channel)
            print('Calculating MICe coefficients...')
            MIC_dict = Parallel_MIC(loaded_dataset, main_channel)
            
            dicts_bin[start] = {'PCC_dict': PCC_dict,'Kendall_dict': Kendall_dict ,'MIC_dict': MIC_dict}

        head = ['channel']
        head.extend(sorted(dicts_bin.keys()))
        PCC_trend_bin = [head]
        Kendall_trend_bin = [head]
        MIC_trend_bin = [head]
        for row in AuxChannels:
            aux_channel = row['name']
            PCC_trend_row_bin = [aux_channel]
            Kendall_trend_row_bin = [aux_channel]
            MIC_trend_row_bin = [aux_channel]
            for start in sorted(dicts_bin.keys()):
                try:
                    PCC_value = dicts_bin[start]['PCC_dict'][aux_channel]
                except KeyError:
                    PCC_value = 'nan'
                try:
                    Kendall_value = dicts_bin[start]['Kendall_dict'][aux_channel]
                except KeyError:
                    Kendall_value = 'nan'
                try:
                    MIC_value = dicts_bin[start]['MIC_dict'][aux_channel]
                except KeyError:
                    MIC_value = 'nan'
                PCC_trend_row_bin.append(PCC_value)
                Kendall_trend_row_bin.append(Kendall_value)
                MIC_trend_row_bin.append(MIC_value)
                        
            PCC_trend_bin.append(PCC_trend_row_bin) 
            Kendall_trend_bin.append(Kendall_trend_row_bin) 
            MIC_trend_bin.append(MIC_trend_row_bin)

        PCC_csv = open('{0}data/PCC_trend_{1}-{2}_{3}-{4}.csv'.format(output_path, int(gst), int(get-gst), main_channel,int(stride)), 'w')
        PCC_csvwriter = csv.writer(PCC_csv)
        for row in PCC_trend_bin:
            PCC_csvwriter.writerow(row)
        PCC_csv.close() 

        Kendall_csv = open('{0}data/Kendall_trend_{1}-{2}_{3}-{4}.csv'.format(output_path, int(gst), int(get-gst), main_channel, int(stride)), 'w')
        Kendall_csvwriter = csv.writer(Kendall_csv)
        for row in Kendall_trend_bin:
            Kendall_csvwriter.writerow(row)
        Kendall_csv.close()
                
        MIC_csv = open('{0}data/MICe_trend_{1}-{2}_{3}-{4}.csv'.format(output_path, int(gst), int(get-gst), main_channel, int(stride)), 'w')
        MIC_csvwriter = csv.writer(MIC_csv)
        for row in MIC_trend_bin:
            MIC_csvwriter.writerow(row)
        MIC_csv.close()                                                  

# Coefficient trend within active segments
def Coefficients_Trend_Segment(output_path, framefiles_path, aux_channels_file_path, segment, gst, get, stride, sample_rate, preprocessing_options, main_channel):
    dicts_bin = dict()
    flag = segment.active
    flaged_segments = list()
    if len(flag) == 1:
        if float(gst) == float(flag[0][0]) and float(get) == float(flag[0][1]):
            segments = np.arange(gst, get, stride)
            for start in segments:
                if flag[0][0] <= start and flag[0][1] >= start+stride and sample_rate * stride > 1000:
                    flaged_segments.append((start, start+stride, 'ON'))
                elif sample_rate * stride < 1000:
                    raise ValueError('These arguments are not vailable if the number of valiables in a set is less than 1 000')
                    sys.exit()
        else:
            if float(gst) == flag[0][0]:
                all_flag = [(flag[0][0],flag[0][1],'ON'),(flag[0][1],float(get),'OFF')]
            elif float(get) == flag[0][1]:
                all_flag = [(float(gst),flag[0][0],'OFF'),(flag[0][0],flag[0][1],'ON')]
            else:
                all_flag = [(float(gst),flag[0][0],'OFF'),(flag[0][0],flag[0][1],'ON'),(flag[0][1],float(get),'OFF')]
            for item in all_flag:
                segments = np.arange(item[0], item[1], stride)
                status = item[2]
                for start in segments:
                    if status == 'ON' and item[0] <= start and item[1] >= start+stride and sample_rate * stride > 1000:
                        flaged_segments.append((start, start+stride, 'ON'))
                    elif status == 'OFF' and item[0] <= start and item[1] >= start+stride and sample_rate * stride > 1000:
                        flaged_segments.append((start, start+stride, 'OFF'))
                    elif sample_rate * stride < 1000:
                        raise ValueError('These arguments are not vailable if the number of valiables in a set is less than 1 000')
    elif len(flag) > 1:
        all_flag = list()
        if float(gst) == float(flag[0][0]) and float(get) == float(flag[-1][1]):
            for i in range(len(flag)):
                all_flag.append((flag[i][0], flag[i][1], 'ON'))
                if i < len(flag)-1:
                    all_flag.append((flag[i][1], flag[i+1][0],'OFF'))
        else:
            if float(gst) != float(flag[0][0]):
                all_flag.append((float(gst), flag[0][0],'OFF'))
            for i in range(len(flag)):
                all_flag.append((flag[i][0], flag[i][1], 'ON'))
                if i < len(flag)-1:
                    all_flag.append((flag[i][1], flag[i+1][0],'OFF'))
            if float(get) != float(flag[-1][1]):
                all_flag.append((flag[-1][1], float(get),'OFF'))
        for item in all_flag:
            segments = np.arange(item[0], item[1], stride)
            status = item[2]
            for start in segments:
                if status == 'ON' and item[0] <= start and item[1] >= start+stride and sample_rate * stride > 1000:
                    flaged_segments.append((start, start+stride, 'ON'))
                elif status == 'OFF' and item[0] <= start and item[1] >= start+stride and sample_rate * stride > 1000:
                    flaged_segments.append((start, start+stride, 'OFF'))
                elif sample_rate * stride < 1000:
                    raise ValueError('These arguments are not vailable if the number of valiables in a set is less than 1 000')

    for flaged_segment in flaged_segments:
        if flaged_segment[2] == 'ON':
            start = flaged_segment[0]
            end = flaged_segment[1]
            print('# Segment[{0}/{1}]: {2} - {3} (stride: {4} seconds)'.format(1+list(flaged_segments).index(flaged_segment), len(flaged_segments), start, end, end-start))
            print('# Flagged semgnet: active')
            cache = GWF_Glue(framefiles_path, start, end)
            AuxChannels = Read_AuxChannels(main_channel, aux_channels_file_path)   
            loaded_dataset = Parallel_Load_data(cache, main_channel, AuxChannels, start, end, sample_rate, preprocessing_options)
            print('Calculating PCC coefficients...')
            PCC_dict = PCC(loaded_dataset, main_channel)
            print('Calculating Kendall coefficients...')
            Kendall_dict = Kendall(loaded_dataset, main_channel)
            print('Calculating MICe coefficients...')
            MIC_dict = Parallel_MIC(loaded_dataset, main_channel)

            dicts_bin[start] = {'PCC_dict': PCC_dict,'Kendall_dict': Kendall_dict ,'MIC_dict': MIC_dict}

        elif flaged_segment[2] == 'OFF':
            start = flaged_segment[0]
            end = flaged_segment[1]
            print('# Segment[{0}/{1}]: {2} - {3} (stride: {4} seconds)'.format(1+list(flaged_segments).index(flaged_segment), len(flaged_segments), start, end, end-start))
            print('# Flaged semgnet: inactive')
            PCC_dict = dict()
            Kendall_dict = dict()
            MIC_dict = dict()
            AuxChannels = Read_AuxChannels(main_channel, aux_channels_file_path)
            for AuxChannel in AuxChannels:
                channel = AuxChannel['name']
                PCC_dict[channel] = 'inactive'
                Kendall_dict[channel] = 'inactive'
                MIC_dict[channel] = 'inactive'
            dicts_bin[start] = {'PCC_dict': PCC_dict,'Kendall_dict': Kendall_dict ,'MIC_dict': MIC_dict}

    head = ['channel']
    head.extend(sorted(dicts_bin.keys()))
    PCC_trend_bin = [head]
    Kendall_trend_bin = [head]
    MIC_trend_bin = [head]
    for row in AuxChannels:
        aux_channel = row['name']
        PCC_trend_row_bin = [aux_channel]
        Kendall_trend_row_bin = [aux_channel]
        MIC_trend_row_bin = [aux_channel]
        for start in sorted(dicts_bin.keys()):
            try:
                PCC_value = dicts_bin[start]['PCC_dict'][aux_channel]
            except KeyError:
                PCC_value = 'nan'
            try:
                Kendall_value = dicts_bin[start]['Kendall_dict'][aux_channel]
            except KeyError:
                Kendall_value = 'nan'
            try:
                MIC_value = dicts_bin[start]['MIC_dict'][aux_channel]
            except KeyError:
                MIC_value = 'nan'
            PCC_trend_row_bin.append(PCC_value)
            Kendall_trend_row_bin.append(Kendall_value)
            MIC_trend_row_bin.append(MIC_value)

            PCC_trend_bin.append(PCC_trend_row_bin) 
            Kendall_trend_bin.append(Kendall_trend_row_bin) 
            MIC_trend_bin.append(MIC_trend_row_bin)

    PCC_csv = open('{0}data/PCC_trend_{1}-{2}_{3}-{4}.csv'.format(output_path, int(gst), int(get-gst), main_channel,int(stride)), 'w')
    PCC_csvwriter = csv.writer(PCC_csv)
    for row in PCC_trend_bin:
        PCC_csvwriter.writerow(row)
    PCC_csv.close() 

    Kendall_csv = open('{0}data/Kendall_trend_{1}-{2}_{3}-{4}.csv'.format(output_path, int(gst), int(get-gst), main_channel, int(stride)), 'w')
    Kendall_csvwriter = csv.writer(Kendall_csv)
    for row in Kendall_trend_bin:
        Kendall_csvwriter.writerow(row)
    Kendall_csv.close()

    MIC_csv = open('{0}data/MICe_trend_{1}-{2}_{3}-{4}.csv'.format(output_path, int(gst), int(get-gst), main_channel, int(stride)), 'w')
    MIC_csvwriter = csv.writer(MIC_csv)
    for row in MIC_trend_bin:
        MIC_csvwriter.writerow(row)
    MIC_csv.close()
