# -*- coding: utf-8 -*-
# Copyright (C) Phil Jung (2020)
#
# This file is port of CAGMon.
#
# CAGMon is the tool that evaluates the dependence between the primary and auxiliary channels of Gravitational-Wave detectors.
#
# CAGMon is following the GNU General Public License version 3. Under this term, you can redistribute and/or modify it.
# See the GNU free software license for more details.

'''conchord : the core utility for making plots.
'''

from cagmon.agrement import *

__author__ = 'Phil Jung <pjjung@nims.re.kr>'

###------------------------------------------### Plot Trend ###-------------------------------------------###     
# Make coefficient trend polts
def Plot_Coefficients_Trend(output_path, gst, get, stride, main_channel, AuxChannels):
    if not output_path.split('/')[-1] == '':
        output_path = output_path + '/'
    csv_path = output_path + 'data/'
    Kendall_csv_path = csv_path + 'Kendall_trend_{0}-{1}_{2}-{3}.csv'.format(int(gst), int(get-gst), main_channel, int(stride)) 
    PCC_csv_path = csv_path + 'PCC_trend_{0}-{1}_{2}-{3}.csv'.format(int(gst), int(get-gst), main_channel, int(stride))
    MIC_csv_path = csv_path + 'MICe_trend_{0}-{1}_{2}-{3}.csv'.format(int(gst), int(get-gst), main_channel, int(stride))
    trend_plots_save_path = output_path + 'plots/Trend/'
    segment_path = output_path + 'segments/FlagSegment.json'
    
    csv_paths = [Kendall_csv_path, PCC_csv_path, MIC_csv_path]
    trend_dict = dict()
    for_median_dict = dict()
    for csv_path in csv_paths:
        if csv_paths.index(csv_path) == 0:
            with open(csv_path, 'r') as csvfile:
                coefficient_name = csv_path.split('/')[-1].split('_')[0]
                reader = list(csv.reader(csvfile))
                for n in range(len(reader)):
                    if n == 0:
                        gps_times = [float(gpstime) for gpstime in reader[n][1:]]              
                    else:
                        channel_name = reader[n][0]
                        coefficient_values = list()
                        for_median_bin = list()
                        for value in reader[n][1:]:
                            if value == 'nan':
                                coefficient_values.append(0)
                                for_median_bin.append(0)
                            elif value == 'inactive':
                                coefficient_values.append(0)
                            elif value != 'nan' and value != 'inactive':
                                coefficient_values.append(float(value))
                                for_median_bin.append(float(value))
                        trend_dict[channel_name] = {coefficient_name:{'x': gps_times,'y':coefficient_values}}   
                        for_median_dict[channel_name] = {coefficient_name:{'median':np.median(for_median_bin)}}
        else:
            with open(csv_path, 'r') as csvfile:
                coefficient_name = csv_path.split('/')[-1].split('_')[0]
                reader = list(csv.reader(csvfile))
                for n in range(len(reader)):
                    if n == 0:
                        gps_times = [float(gpstime) for gpstime in reader[n][1:]]               
                    else:
                        channel_name = reader[n][0]
                        coefficient_values = list()
                        for_median_bin = list()
                        for value in reader[n][1:]:
                            if value == 'nan':
                                coefficient_values.append(0)
                                for_median_bin.append(0)
                            elif value == 'inactive':
                                coefficient_values.append(0)
                            elif value != 'nan' and value != 'inactive':
                                coefficient_values.append(float(value))
                                for_median_bin.append(float(value))
                        trend_dict[channel_name].update({coefficient_name:{'x': gps_times,'y':coefficient_values}})
                        for_median_dict[channel_name].update({coefficient_name:{'median':np.median(for_median_bin)}})
   
    for row in AuxChannels:
        channel_name = row['name']
        fig = plt.figure(figsize=(12,6))
        plot = GridSpec(2, 1, height_ratios=[30, 1])
        ax = fig.add_subplot(plot[0])
        ax2 = fig.add_subplot(plot[1])
        max_values = list()
        for coefficient in ['MICe', 'PCC', 'Kendall']:
            x = trend_dict[channel_name][coefficient]['x']
            x_ = [float(i)-float(x[0]) for i in x]
            y = trend_dict[channel_name][coefficient]['y']
            median = for_median_dict[channel_name][coefficient]['median']
            median_line = [median for i in range(len(x_))]
            start_datetime = tconvert(x[0]).strftime('%Y-%m-%d %H:%M:%S')
            if 0 < x_[-1] < 60:
                xlabel = 'Time [seconds] from {0} UTC ({1})'.format(start_datetime, x[0]) 
                xscale = 's'
            elif 60 <= x_[-1] < 3600:
                xlabel = 'Time [minutes] from {0} UTC ({1})'.format(start_datetime, x[0])           
                x_ = np.array(x_)/60
                xscale = 'm'
            elif 3600 <= x_[-1] < 3600*24:
                xlabel = 'Time [hours] from {0} UTC ({1})'.format(start_datetime, x[0])           
                x_ = np.array(x_)/3600
                xscale = 'h'
            elif x_[-1] <= 3600*24:
                xlabel = 'Time [days] from {0} UTC ({1})'.format(start_datetime, x[0])           
                x_ = np.array(x_)/(3600*24)
                xscale = 'd'
                
            if coefficient == 'MICe':
                line_color = 'red'
                y_max = max(y)
                x_point = x_[y.index(y_max)]
                max_values.append(float(y_max))
                ax.plot(x_point, y_max, marker='*', color='gold', markersize=15)
                ax.plot(x_, y, label='MICe', color=line_color)
                ax.plot(x_, median_line, label='median', linestyle='dashed', color=line_color)
            elif coefficient == 'PCC':
                y_max_pcc = max(y)
                x_point_pcc = x_[y.index(y_max_pcc)]
                max_values.append(float(y_max_pcc))
                line_color = 'green'
                ax.plot(x_, y, label='PCC', color=line_color)
                ax.plot(x_, median_line, label='median', linestyle='dashed', color=line_color)
            elif coefficient == 'Kendall':
                y_max_kendall = max(y)
                x_point_kendall = x_[y.index(y_max_kendall)]
                max_values.append(float(y_max_kendall))
                line_color = 'blue'
                ax.plot(x_, y, label='Kendall', color=line_color)
                ax.plot(x_, median_line, label='median', linestyle='dashed', color=line_color)
                
        y_axis_max = max(max_values)
        if y_axis_max*1.2 >= 1:
            ax.set_ylim([0., 1.])
        else:
            ax.set_ylim([0., y_axis_max*1.2])
            
        ax.set_xlim([x_[0], x_[-1]])
        ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
        ax.set_title('Coefficients Trend {0} (stride: {1} seconds)'.format(channel_name, stride))
        ax.set_ylabel('Coefficient Value')
        ax2.set_xlabel(xlabel)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=10)

        segment = DataQualityFlag.read(segment_path)
        flag = segment.active
        flaged_segments = list()
        if len(flag) == 0:
            flaged_segments.append([gst, get,'darkred'])
        elif len(flag) == 1:
            if int(gst) == int(flag[0][0]) and int(get) == int(flag[0][1]):
                flaged_segments.append([flag[0][0], flag[0][1], 'limegreen'])
            else:
                if int(gst) == flag[0][0]:
                    flaged_segments.append([flag[0][0],flag[0][1],'limegreen'])
                    flaged_segments.append([flag[0][1],int(get),'darkred'])
                elif int(get) == flag[0][1]:
                    flaged_segments.append([int(gst),flag[0][0],'darkred'])
                    flaged_segments.append([flag[0][0],flag[0][1],'limegreen'])
                else:
                    flaged_segments.append([int(gst),flag[0][0],'darkred'])
                    flaged_segments.append([flag[0][0],flag[0][1],'limegreen'])
                    flaged_segments.append([flag[0][1],int(get),'darkred'])
        elif len(flag) > 1:
            if int(gst) == int(flag[0][0]) and int(get) == int(flag[-1][1]):
                for i in range(len(flag)):
                    flaged_segments.append([flag[i][0], flag[i][1], 'limegreen'])
                    if i < len(flag)-1:
                        flaged_segments.append([flag[i][1], flag[i+1][0],'darkred'])
            else:
                if int(gst) != int(flag[0][0]):
                    flaged_segments.append([int(gst), flag[0][0],'darkred'])
                for i in range(len(flag)):
                    flaged_segments.append([flag[i][0], flag[i][1], 'limegreen'])
                    if i < len(flag)-1:
                        flaged_segments.append([flag[i][1], flag[i+1][0],'darkred'])
                if int(get) != int(flag[-1][1]):
                    flaged_segments.append([flag[-1][1], int(get),'darkred']) 

        ax.set_xticklabels([])
        ax.set_xticks([])
        widths = list()
        starts = list()
        colors = list()
        left = 0
        for start, end, color in flaged_segments:
            widths.append(int(end-start))
            starts.append(int(left))
            left += int(end-start)
            colors.append(color)  
        if xscale == 's':
            widths_ = np.array(widths)       
            starts_ = np.array(starts)          
        elif xscale == 'm': 
            widths_ = np.array(widths)/60.        
            starts_ = np.array(starts)/60.
        elif xscale == 'h':
            widths_ = np.array(widths)/3600.        
            starts_ = np.array(starts)/3600.           
        elif xscale == 'd':
            widths_ = np.array(widths)/(3600.*24)       
            starts_ = np.array(starts)/(3600.*24)  

        for i in range(len(widths_)):
            if colors[i] == 'darkred':
                ax2.barh(0, widths_[i], left=starts_[i], height=.06, color=colors[i], align='center', linewidth=0)
            elif colors[i] == 'limegreen':
                ax2.barh(0, widths_[i], left=starts_[i], height=.10, color=colors[i], align='center', linewidth=0)
        ax2.set_ylabel('Active', rotation='horizontal', ha='right', va="center")
        ax2.set_xlim(0, x_[-1])
        ax2.grid(False)
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        plt.subplots_adjust(hspace=0.05)
        
        plt.savefig('{0}/Coefficients-Trend_{1}-{2}_{3}_{4}.png'.format(trend_plots_save_path, int(gst), int(get-gst), channel_name, int(stride)))
        print('Saved Trend plot: {}'.format(channel_name))

# Make coefficient distribution plots regardless of segment information  
def Plot_Distribution_Trend(output_path, gst, get, main_channel, stride, ctype):
    if not output_path.split('/')[-1] == '':
        output_path = output_path + '/'
    data_path = output_path+'data/{0}_trend_{1}-{2}_{3}-{4}.csv'.format(ctype, int(gst), int(get-gst), main_channel, int(stride))
    distribution_trend_plot_save_path = output_path + 'plots/Trend/'
    segment_path = output_path + 'segments/FlagSegment.json'
    
    number_up9 = list()
    number_up8 = list()
    number_up7 = list()
    number_up6 = list()
    number_up5 = list()
    number_up4 = list()
    number_up3 = list()
    number_up2 = list()
    number_up1 = list()
    number_up0 = list()

    rawdata = list()
    with open(data_path, 'r') as raw:
        reader = csv.DictReader(raw)
        for line in reader:
            rawdata.append(line)

    gps_times = list()
    for key in rawdata[0].keys():
        if not key == 'channel':
            gps_times.append(key)
    gps_times = sorted(gps_times)
    x = [float(i) for i in gps_times]

    for gpstime in gps_times:
        up9 = list()
        up8 = list()
        up7 = list()
        up6 = list()
        up5 = list()
        up4 = list()
        up3 = list()
        up2 = list()
        up1 = list()
        up0 = list()
        tmp_bin = [float(i[gpstime]) for i in rawdata]
        for value in tmp_bin:
            if value >= 0.9:
                up9.append(value)
            elif 0.9> value >= 0.8:
                up8.append(value)
            elif 0.8> value >= 0.7:
                up7.append(value)
            elif 0.7> value >= 0.6:
                up6.append(value)
            elif 0.6> value >= 0.5:
                up5.append(value)
            elif 0.5> value >= 0.4:
                up4.append(value)
            elif 0.4> value >= 0.3:
                up3.append(value)
            elif 0.3> value >= 0.2:
                up2.append(value)
            elif 0.2> value >= 0.1:
                up1.append(value)
            else:
                up0.append(value)
        number_up9.append(len(up9))
        number_up8.append(len(up8))
        number_up7.append(len(up7))
        number_up6.append(len(up6))
        number_up5.append(len(up5))
        number_up4.append(len(up4))
        number_up3.append(len(up3))
        number_up2.append(len(up2))
        number_up1.append(len(up1))
        number_up0.append(len(up0))

    bar_dict = dict()
    number_ups = [number_up9, number_up8, number_up7, number_up6, number_up5, number_up4, number_up3, number_up2, number_up1, number_up0]

    count = 0
    while count < 10:
        if count == 0:
            bottom = np.array([0 for i in range(len(x))])
        else:
            bottom += np.array(number_ups[count-1])
        bar_dict[count] = {'y': number_ups[count],'bottom':list(bottom), 'label':'>={}'.format(round(0.9-count/10.,1))}
        count += 1

    x_ = [float(i)-float(gps_times[0]) for i in gps_times]
    start_datetime = tconvert(x[0]).strftime('%Y-%m-%d %H:%M:%S')
    if 0 < x_[-1] < 60:
        xlabel = 'Time [seconds] from {0} UTC ({1})'.format(start_datetime, x[0])
        xscale = 's'
    elif 60 <= x_[-1] < 3600:
        xlabel = 'Time [minutes] from {0} UTC ({1})'.format(start_datetime, x[0])           
        x_ = np.array(x_)/60
        xscale = 'm'
    elif 3600 <= x_[-1] < 3600*24:
        xlabel = 'Time [hours] from {0} UTC ({1})'.format(start_datetime, x[0])           
        x_ = np.array(x_)/3600
        xscale = 'h'
    elif x_[-1] <= 3600*24:
        xlabel = 'Time [days] from {0} UTC ({1})'.format(start_datetime, x[0])           
        x_ = np.array(x_)/(3600*24)
        xscale = 'd'
    
    if ctype == 'MICe':
        color_dict = {0:'black', 1:'darkred',2:'brown',3:'red',4:'coral',5:'orange',6:'darkorange',7:'grey',8:'darkgrey',9:'lightgrey'}
    elif ctype == 'PCC':
        color_dict = {0:'black', 1:'darkgreen',2:'green',3:'limegreen',4:'yellowgreen',5:'yellow',6:'gold',7:'grey',8:'darkgrey',9:'lightgrey'}
    elif ctype == 'Kendall':
        color_dict = {0:'black', 1:'navy',2:'blue',3:'deepskyblue',4:'violet',5:'magenta',6:'darkmagenta',7:'grey',8:'darkgrey',9:'lightgrey'}
    
    fig = plt.figure(figsize=(12,6))
    plot = GridSpec(2, 1, height_ratios=[30, 1])
    ax = fig.add_subplot(plot[0])
    ax2 = fig.add_subplot(plot[1])
    for condition in bar_dict.keys():
        ax.bar(x_, bar_dict[condition]['y'],width=(x_[1]-x_[0]) ,label=bar_dict[condition]['label'], color=color_dict[condition], bottom=bar_dict[condition]['bottom'],linewidth=0, align='edge')
    ax.set_ylim(0,len(rawdata))
    ax.set_xlim(0, x_[-1])
    ax.set_ylabel('Number of channels')
    ax2.set_xlabel(xlabel)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=10)
    ax.set_title('Coeffiient distribution trend of {0} (stride: {1} seconds)'.format(ctype, stride))

    segment = DataQualityFlag.read(segment_path)
    flag = segment.active
    flaged_segments = list()
    if len(flag) == 0:
        flaged_segments.append([gst, get,'darkred'])
    elif len(flag) == 1:
        if int(gst) == int(flag[0][0]) and int(get) == int(flag[0][1]):
            flaged_segments.append([flag[0][0], flag[0][1], 'limegreen'])
        else:
            if int(gst) == flag[0][0]:
                flaged_segments.append([flag[0][0],flag[0][1],'limegreen'])
                flaged_segments.append([flag[0][1],int(get),'darkred'])
            elif int(get) == flag[0][1]:
                flaged_segments.append([int(gst),flag[0][0],'darkred'])
                flaged_segments.append([flag[0][0],flag[0][1],'limegreen'])
            else:
                flaged_segments.append([int(gst),flag[0][0],'darkred'])
                flaged_segments.append([flag[0][0],flag[0][1],'limegreen'])
                flaged_segments.append([flag[0][1],int(get),'darkred'])
    elif len(flag) > 1:
        if int(gst) == int(flag[0][0]) and int(get) == int(flag[-1][1]):
            for i in range(len(flag)):
                flaged_segments.append([flag[i][0], flag[i][1], 'limegreen'])
                if i < len(flag)-1:
                    flaged_segments.append([flag[i][1], flag[i+1][0],'darkred'])
        else:
            if int(gst) != int(flag[0][0]):
                flaged_segments.append([int(gst), flag[0][0],'darkred'])
            for i in range(len(flag)):
                flaged_segments.append([flag[i][0], flag[i][1], 'limegreen'])
                if i < len(flag)-1:
                    flaged_segments.append([flag[i][1], flag[i+1][0],'darkred'])
            if int(get) != int(flag[-1][1]):
                flaged_segments.append([flag[-1][1], int(get),'darkred']) 

    ax.set_xticklabels([])
    ax.set_xticks([])
    widths = list()
    starts = list()
    colors = list()
    left = 0
    for start, end, color in flaged_segments:
        widths.append(int(end-start))
        starts.append(int(left))
        left += int(end-start)
        colors.append(color)  
    if xscale == 's':
        widths_ = np.array(widths)       
        starts_ = np.array(starts)          
    elif xscale == 'm': 
        widths_ = np.array(widths)/60.        
        starts_ = np.array(starts)/60.
    elif xscale == 'h':
        widths_ = np.array(widths)/3600.        
        starts_ = np.array(starts)/3600.           
    elif xscale == 'd':
        widths_ = np.array(widths)/(3600.*24)       
        starts_ = np.array(starts)/(3600.*24)  

    for i in range(len(widths_)):
        if colors[i] == 'darkred':
            ax2.barh(0, widths_[i], left=starts_[i], height=.06, color=colors[i], align='center', linewidth=0)
        elif colors[i] == 'limegreen':
            ax2.barh(0, widths_[i], left=starts_[i], height=.10, color=colors[i], align='center', linewidth=0)
    ax2.set_ylabel('Active', rotation='horizontal', ha='right', va="center")
    ax2.set_xlim(0, x_[-1])
    ax2.grid(False)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    plt.subplots_adjust(hspace=0.05) 
    
    plt.savefig('{0}/{1}_Coefficient-Distribution-Trend_{2}-{3}_{4}_{5}.png'.format(distribution_trend_plot_save_path, ctype, int(gst), int(get-gst), main_channel, int(stride)))
    print('Saved Distribution Trend plot: {}'.format(ctype))


###------------------------------------------### Scatter and OmegaScan ###-------------------------------------------###
def Scatter(framefiles_path, output_path, main_channel, aux_channel, gst, get, marked_gst, marked_get, sample_rate, preprocessing_options):
    if not output_path.split('/')[-1] == '':
        output_path = output_path + '/'
    cache = GWF_Glue(framefiles_path, gst, get)

    y = np.array(ReadTimeseries(cache, main_channel, marked_gst, marked_get, sample_rate, preprocessing_options))
    x = np.array(ReadTimeseries(cache, aux_channel, marked_gst, marked_get, sample_rate, preprocessing_options))
    x = x/((x**2).sum())**(0.5)
    y = y/((y**2).sum())**(0.5)
    plt.figure(figsize=(12,6))
    plt.scatter(x,y)
    plt.title('Scatter plot ({0}-{1})'.format(marked_gst, marked_get-marked_gst))
    plt.ylabel('{0} ({1})'.format(main_channel, 'normalized'))
    plt.xlabel('{0} ({1})'.format(aux_channel, 'normalized'))
    plt.savefig('{0}plots/Scatter/Scatter_{1}-{2}_{3}.png'.format(output_path, int(gst), int(get-gst), aux_channel))
    print('Saved Scatter plot: {}'.format(aux_channel))


def OmegaScan(framefiles_path, output_path, channel, gst, get, marked_gst, marked_get, preprocessing_options):
    if not output_path.split('/')[-1] == '':
        output_path = output_path + '/'
    reading_gst = marked_gst + (marked_get-marked_gst)*0.5 - 17
    reading_get = marked_gst + (marked_get-marked_gst)*0.5 + 17
    
    cache = GWF_Glue(framefiles_path, reading_gst, reading_get)
    data = ReadTimeseries(cache, channel, reading_gst, reading_get, None, preprocessing_options) 
    
    qspecgram = data.q_transform(qrange=(4, 150), gps=None, search=0.5, tres=0.001, fres=0.5, norm='median', outseg=None, whiten=True, fduration=1, highpass=None)
    q_plot = qspecgram.crop( marked_gst, marked_get).plot(figsize=[12,6])
    qax = q_plot.gca()
    qax.set_xscale('seconds')
    qax.set_yscale('log')
    qax.set_xlim(marked_gst, marked_get)
    maxy = float(str(qspecgram.yindex[-1]).split(' ')[0])
    miny = float(str(qspecgram.yindex[0]).split(' ')[0])
    qax.set_xlabel('Time [Seconds]{0} UTC ({1})'.format(tconvert(marked_gst).strftime('%Y-%m-%d %H:%M:%S'), marked_gst))
    qax.set_ylim(miny, maxy)
    qax.set_ylabel('Frequency [Hz]')
    qax.set_title('Q-transformation ({})'.format(channel))
    qax.grid(False)
    q_plot.add_colorbar(cmap='gnuplot', label='Normalized energy')
    q_plot.savefig('{0}plots/OmegaScan/OmegaScan_{1}-{2}_{3}.png'.format(output_path, int(gst), int(get-gst), channel))
    print('Saved OmegaScan plot: {}'.format(channel))

def Spectrogram(framefiles_path, output_path, channel, gst, get, marked_gst, marked_get, preprocessing_options):
    if not output_path.split('/')[-1] == '':
        output_path = output_path + '/'
    cache = GWF_Glue(framefiles_path, reading_gst, reading_get)
    data = ReadTimeseries(cache, channel, marked_gst, marked_get, None, preprocessing_options) 
    spectro = data.spectrogram(2) ** (1/2.)
    normalised = spectro.ratio('median')
    plot = normalised.plot(figsize=(12,6))
    ax = plot.gca()
    ax.set_yscale('log')
    ax.set_xscale('auto-gps')
    ax.set_xlim(marked_gst, marked_get)
    maxy = float(str(normalised.yindex[-1]).split(' ')[0])
    miny = float(str(normalised.yindex[0]).split(' ')[0])
    ax.set_ylim(miny, maxy)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('Spectrogram ({})'.format(channel))
    ax.colorbar(cmap='gnuplot', norm='log', label='Relative amplitude')
    plot.savefig('{0}plots/OmegaScan/OmegaScan_{1}-{2}_{3}.png'.format(output_path, int(gst), int(get-gst), channel))
    print('Saved Spectrogram plot: {}'.format(channel))
