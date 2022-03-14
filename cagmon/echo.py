# -*- coding: utf-8 -*-
# Copyright (C) Phil Jung (2020)
#
# This file is port of CAGMon.
#
# CAGMon is the tool that evaluates the dependence between the primary and auxiliary channels of Gravitational-Wave detectors.
#
# CAGMon is following the GNU General Public License version 3. Under this term, you can redistribute and/or modify it.
# See the GNU free software license for more details.

'''echo : the utility for making HTML summary page.
'''

import os
from gwpy.time import tconvert

__author__ = 'Phil Jung <pjjung@nims.re.kr>'

#---------------------------------------------------------------------------------------------------------#

def html_head():
    html_head = '''<!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8"><link rel="stylesheet" href="./css/style.css">
    </head>
    <body>
    <div class="row Head" id="head_row">
      <div class="cell" id="head_cell">
        <div id="head_title">CAGMon: Correlation Analysis based on Glitch Monitoring
        </div>
      </div>
    </div>
    '''
    
    return html_head

def html_configuration(gst, get, coefficients_trend_stride, filter_type, freq1, freq2, main_channel, mic_alpha, mic_c, sample_rate):
    options = ''
    if filter_type == 'lowpass':
        options += ' ,lowpass filter: {}Hz'.format(freq1)
    elif filter_type == 'highpass':
        options += ', highpass filter: {}Hz'.format(freq1)
    elif filter_type == 'bandpass':
        options += ', bandpass filter: {}-{}Hz'.format(freq1, freq2)
        
    html_configuration = '''
    <div class="row Configuration" id="row">
      <div class="cell" id="cell">
        <div id="sub_title">Configuration
        </div>
        <hr id="line"/>
        <div id="config_text">
            <span>{0} - {1} </span>
            <div id="config_text_detail">
              <span>Active segment: <a draggable="true" href="./segments/FlagSegment.txt" target="_blank" rel="noopener">txt</a> <a draggable="true" href="./segments/FlagSegment.json" target="_blank" rel="noopener">json</a> </span>
            </div>
            <div id="config_text_detail">
              <span>stride: {2} seconds, sample rate: {8}Hz, data size: {3}{4}</span>
            </div>
            <div id="config_text_detail">
              <span>main channel: {5}</span>
            </div>
            <div id="config_text_detail">
              <span>MICe Alpha: {6}, MICe c: {7}</span>
            </div>
        </div>
      </div>
    </div>
    '''.format(gst, get, coefficients_trend_stride, int(coefficients_trend_stride*sample_rate), options, main_channel, mic_alpha, mic_c, sample_rate)
    
    return html_configuration

def html_summary(gst, get, coefficients_trend_stride, main_channel, MIC_maxvalues, PCC_maxvalues, Kendall_maxvalues, sorted_MIC_maxvalues):
    html_summary = '''
    <div class="row Sammury" id="row">
      <div class="cell" id="cell">
        <div id="sub_title">Summary
        </div>
        <hr id="line"/>
        <div id="slider">
          <div class="gjs-lory-frame" id="slider_frame">
            <div class="gjs-lory-slides" id="slider_slides">
              <div class="gjs-lory-slide" id="slider_slide">
                <img id="slide_image" src="{0}"/>
              </div>
              <div class="gjs-lory-slide" id="slider_slide">
                <img id="slide_image" src="{1}"/>
              </div>
              <div class="gjs-lory-slide" id="slider_slide">
                <img id="slide_image" src="{2}"/>
              </div>
            </div>
          </div>
          <span class="gjs-lory-prev" id="slider_left_arrow"><svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" viewBox="0 0 501.5 501.5">
            <g>
              <path fill="#2E435A" d="M302.67 90.877l55.77 55.508L254.575 250.75 358.44 355.116l-55.77 55.506L143.56 250.75z">
              </path>
            </g>
            </svg></span>
          <span class="gjs-lory-next" id="slider_right_arrow"><svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" viewBox="0 0 501.5 501.5">
            <g>
              <path fill="#2E435A" d="M199.33 410.622l-55.77-55.508L247.425 250.75 143.56 146.384l55.77-55.507L358.44 250.75z">
              </path>
            </g>
            </svg></span>
        </div>
        <div id="summary_legend">*LTMV: lower than the median value
        </div>
        <hr id="line"/>
        <div>
          <table id="summary_table">
            <tbody>
              <tr>
                <td id="table_cell_channel">Channel
                </td>
                <td id="table_cell_values">MICe
                </td>
                <td id="table_cell_values">MICe median
                </td>
                <td id="table_cell_values">PCC
                </td>
                <td id="table_cell_values">PCC median
                </td>
                <td id="table_cell_values">Kendall
                </td>
                <td id="table_cell_values">Kendall median
                </td>
                <td id="table_cell_segment">segment
                </td>
              </tr>
              <tr>
              </tr>
            </tbody>
          </table>
          {3}
        </div>
        <hr id="table_line"/>
      </div>
    </div>
    '''

    summary_table = '''
          <hr id="table_line"/>
          <table id="summary_table">
            <tbody>
              <tr>
                <td id="table_cell_channel">{0}
                </td>
                <td id="table_cell_values">{1}
                </td>
                <td id="table_cell_values">{2}
                </td>
                <td id="table_cell_values">{3}
                </td>
                <td id="table_cell_values">{4}
                </td>
                <td id="table_cell_values">{5}
                </td>
                <td id="table_cell_values">{6}
                </td>
                <td id="table_cell_segment">{7}
                </td>
              </tr>
              <tr>
              </tr>
            </tbody>
          </table>
    '''

    tables = ''''''
    for max_MIC_info in sorted_MIC_maxvalues:
        channel = max_MIC_info[0]
        start = max_MIC_info[1]
        MIC = round(MIC_maxvalues[channel]['values'][start],2)
        MIC_median = round(MIC_maxvalues[channel]['median'],2)
        try:
            PCC_median = round(PCC_maxvalues[channel]['median'],2)
        except KeyError:
            PCC_median = 0.
        try:
            Kendall_median = round(Kendall_maxvalues[channel]['median'],2)
        except KeyError:
            Kendall_median = 0.
        try:
            PCC = round(PCC_maxvalues[channel]['values'][start],2)
        except KeyError:
            PCC = 'LTMV'
        try:
            Kendall = round(Kendall_maxvalues[channel]['values'][start],2)
        except KeyError:
            Kendall = 'LTMV'
        segment = '{0} - {1}'.format(start, start+coefficients_trend_stride)
        table = summary_table.format(channel, MIC, MIC_median, PCC, PCC_median, Kendall, Kendall_median, segment)
        tables += table

    MIC_coefficient_contribution_plot = './plots/Trend/MICe_Coefficient-Distribution-Trend_{0}-{1}_{2}_{3}.png'.format(int(gst), int(get-gst), main_channel, int(coefficients_trend_stride))
    PCC_coefficient_contribution_plot = './plots/Trend/PCC_Coefficient-Distribution-Trend_{0}-{1}_{2}_{3}.png'.format(int(gst), int(get-gst), main_channel, int(coefficients_trend_stride))
    Kendall_coefficient_contribution_plot = './plots/Trend/Kendall_Coefficient-Distribution-Trend_{0}-{1}_{2}_{3}.png'.format(int(gst), int(get-gst), main_channel, int(coefficients_trend_stride))
    
    html_summary = html_summary.format(MIC_coefficient_contribution_plot, PCC_coefficient_contribution_plot, Kendall_coefficient_contribution_plot, tables)

    return html_summary


def html_details(output_path, gst, get, coefficients_trend_stride, main_channel, MIC_maxvalues, PCC_maxvalues, Kendall_maxvalues, sorted_MIC_maxvalues):
    html_details_subtitle = '''
    <div class="row Details" id="row">
      <div class="cell" id="cell">
        <div id="sub_title">Details
        </div>
        <hr id="line"/>
      </div>
    </div>
    '''

    html_details_box ='''
    <div class="row Details" id="row_detail">
      <div class="cell" id="cell_detail">
        <div id="detail_contents">
          <span id="detail_text">Datetime: {12}</span>
          <div id="detail_text">GPS time: {7}
          </div>
          <div id="detail_text">Channel: {0}
          </div>
          <div id="detail_text">MICe: {1}
          </div>
          <div id="detail_text">PCC: {2}
          </div>
          <div id="detail_text">Kendall&#039;s tau: {3}
          </div>
          <div id="detail_text">Median of MICe: {4}
          </div>
          <div id="detail_text">Median of PCC: {5}
          </div>
          <div id="detail_text">Median of Kendall: {6}
          </div>
          <div id="detail_text">Other MICe values: <a draggable="true" href="{8}" target="_blank" rel="noopener">txt</a>
          </div>
          <div id="detail_text">Other PCC values: <a draggable="true" href="{9}" target="_blank" rel="noopener">txt</a>
          </div>
          <div id="detail_text">Other Kendall values: <a draggable="true" href="{10}" target="_blank" rel="noopener">txt</a>
          </div>
        </div>
    </div>
    <div class="cell" id="cell_image">
      <div id="slider">
        <img id="slide_image" src="{11}"/>
      </div>
    </div>
    </div>
    '''

    for channel in MIC_maxvalues.keys():
        txt_bin = list()
        for gps_time in sorted(MIC_maxvalues[channel]['values'].keys()):
            value = MIC_maxvalues[channel]['values'][gps_time]
            txt_bin.append('{0} {1}'.format(gps_time, value))
        f = open('{0}data/MICe_{1}-{2}_{3}.txt'.format(output_path, int(gst), int(get-gst), channel), 'w')
        f.write('\n'.join(txt_bin))
        f.close()
    for channel in PCC_maxvalues.keys():
        txt_bin = list()
        for gps_time in sorted(PCC_maxvalues[channel]['values'].keys()):
            value = PCC_maxvalues[channel]['values'][gps_time]
            txt_bin.append('{0} {1}'.format(gps_time, value))
        f = open('{0}data/PCC_{1}-{2}_{3}.txt'.format(output_path, int(gst), int(get-gst), channel), 'w')
        f.write('\n'.join(txt_bin))
        f.close()
    for channel in Kendall_maxvalues.keys():
        txt_bin = list()
        for gps_time in sorted(Kendall_maxvalues[channel]['values'].keys()):
            value = Kendall_maxvalues[channel]['values'][gps_time]
            txt_bin.append('{0} {1}'.format(gps_time, value))
        f = open('{0}data/Kendall_{1}-{2}_{3}.txt'.format(output_path, int(gst), int(get-gst), channel), 'w')
        f.write('\n'.join(txt_bin))
        f.close()

    details = ''''''
    for max_MIC_info in sorted_MIC_maxvalues:
        channel = max_MIC_info[0]
        start = max_MIC_info[1]
        MIC = round(MIC_maxvalues[channel]['values'][start],2)
        MIC_median = round(MIC_maxvalues[channel]['median'],2)
        try:
            PCC_median = round(PCC_maxvalues[channel]['median'],2)
        except KeyError:
            PCC_median = 0.
        try:
            Kendall_median = round(Kendall_maxvalues[channel]['median'],2)
        except KeyError:
            Kendall_median = 0.
        try:
            PCC = round(PCC_maxvalues[channel]['values'][start],2)
        except KeyError:
            PCC = 'LTMV'
        try:
            Kendall = round(Kendall_maxvalues[channel]['values'][start],2)
        except KeyError:
            Kendall = 'LTMV'
        datetime = '{0}-{1}'.format(tconvert(start).strftime('%Y-%m-%d %H:%M:%S'), coefficients_trend_stride)
        segment = '{0} - {1}'.format(start, start+coefficients_trend_stride)

        trend_plot = './plots/Trend/Coefficients-Trend_{0}-{1}_{2}_{3}.png'.format(int(gst), int(get-gst), channel, int(coefficients_trend_stride))
        
        MIC_filelink = './data/MICe_{0}-{1}_{2}.txt'.format(int(gst), int(get-gst), channel)
        PCC_filelink = './data/PCC_{0}-{1}_{2}.txt'.format(int(gst), int(get-gst), channel)
        Kendall_filelink = './data/Kendall_{0}-{1}_{2}.txt'.format(int(gst), int(get-gst), channel)
        box = html_details_box.format(channel, MIC, PCC, Kendall, MIC_median, PCC_median, Kendall_median, segment, MIC_filelink, PCC_filelink, Kendall_filelink, trend_plot, datetime)
        details += box
        
    html_details = html_details_subtitle + details
    
    return html_details

def html_foot():
    html_foot = '''
    <div class="row Foot" id="foot_row">
      <div class="cell" id="foot_cell">
        <div id="foot_text">Designed by Phil Jung in Korea Gravitational Wave Group (KGWG) </div>
        <div id="foot_text">e-mail: pjjung@nims.re.kr </div>
      </div>
    </div>
    '''
    return html_foot

def html_script():
    html_script = '''
    <script>var items = document.querySelectorAll('#slider');
      for (var i = 0, len = items.length; i < len; i++) {
        (function(){
          var e=this,t="https://cdnjs.cloudflare.com/ajax/libs/lory.js/2.3.4/lory.min.js",l=["0","false"],s="";
          s="true"==s?1:parseInt(s,10);
          var a={
            slidesToScroll:parseInt("1",10),enableMouseEvents:l.indexOf("")>=0?0:1,infinite:!isNaN(s)&&s,rewind:!(l.indexOf("")>=0),slideSpeed:parseInt("300",10),rewindSpeed:parseInt("600",10),snapBackSpeed:parseInt("200",10),ease:"ease",classNameFrame:"gjs-lory-frame",classNameSlideContainer:"gjs-lory-slides",classNamePrevCtrl:"gjs-lory-prev",classNameNextCtrl:"gjs-lory-next"}
          ,r=function(){
            window.sliderLory=lory(e,a)};
          if("undefined"==typeof lory){
            var n=document.createElement("script");
            n.src=t,n.onload=r,document.head.appendChild(n)}
          else r()}
         .bind(items[i]))();
      }
    </script>
    </body>
    <html>
    '''

    return html_script

def css_text():
    css_text = '''
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
    }
    .row{
      display:flex;
      justify-content:flex-start;
      align-items:stretch;
      flex-wrap:nowrap;
      padding:10px;
    }
    .cell{
      min-height:75px;
      flex-grow:1;
      flex-basis:100%;
    }
    #head_row{
      background-image:linear-gradient(#fdb12a,#fdb12a);
      background-repeat:repeat;
      background-position:left top;
      background-attachment:scroll;
      background-size:auto;
      margin:0 0 25px 0;
    }
    #head_cell{
      margin:0 10% 0 10%;
      min-height:auto;
    }
    #head_title{
      padding:10px;
      font-family:Helvetica, sans-serif;
      letter-spacing:0;
      font-size:30px;
      position:relative;
    }
    #foot_row{
      background-image:linear-gradient(#6f6e6c,#6f6e6c);
      background-repeat:repeat;
      background-position:left top;
      background-attachment:scroll;
      background-size:auto;
      margin:25px 0 0 0;
    }
    #foot_cell{
      margin:0 10% 0 10%;
      min-height:auto;
    }
    #row{
      margin:0 10% 0 10%;
    }
    #cell{
      margin:0 0 0 0;
    }
    #line{
      opacity:0.5;
      margin:0 0 8px 0;
    }
    #sub_title{
      padding:10px;
      font-family:Helvetica, sans-serif;
      letter-spacing:0;
      font-size:25px;
      position:relative;
    }
    #summary_legend{
      padding:2px;
      text-align:right;
    }

    #text{
      padding:10px;
      font-family:Helvetica, sans-serif;
      font-size:18px;
    }
    #config_text{
      padding:15px;
      font-family:Helvetica, sans-serif;
      font-size:21px;
    }
    #config_text_detail{
      line-height:150%;
    }

    #cell_detail{
      flex-basis:30%;
    }
    #cell_image{
      flex-basis:70%;
    }
    #row_detail{
      margin:0 11% 0.5% 11%;
      border:1px solid #cacaca;
      border-radius:5px 5px 5px 5px;
    }
    #summary_table{
      width:100%;
    }
    #table_cell_channel{
      width:30%;
      font-family:Helvetica, sans-serif;
      font-size:18px;
      padding:1px 1px 1px 1%;
      text-decoration:none;
    }
    #table_cell_values{
      width:8.33%;
      font-family:Helvetica, sans-serif;
      font-size:18px;
      padding:1px 1px 1px 1%;
    }
    #table_cell_segment{
      width:20%;
      font-family:Helvetica, sans-serif;
      font-size:18px;
      padding:1px 1px 1px 1%;
    }
    #table_line{
      opacity:0.2;
      margin:0 0 8px 0;
    }
    #slider{
      position:relative;
      width:auto;
    }
    #slider_frame{
      width:88%;
      margin:0 auto;
      position:relative;
      overflow:hidden;
      white-space:nowrap;
    }
    #slider_slides{
      display:inline-block;
      transition-delay:1ms;
    }
    #slider_slide{
      display:inline-block;
      position:relative;
      color:#fff;
      width:100%;
      margin-right:0px;
      vertical-align:top;
      min-height:130px;
      white-space:normal;
      background-color:rgba(0, 0, 0, 0.1);
      background-image:linear-gradient(#ffffff,#ffffff);
      background-repeat:repeat;
      background-position:left top;
      background-attachment:scroll;
      background-size:auto;
    }
    #slide_image{
      color:black;
      width:100%;
    }
    #slider_left_arrow{
      position:absolute;
      display:block;
      cursor:pointer;
      top:50%;
      left:0;
      margin-top:-25px;
    }
    #slider_right_arrow{
      position:absolute;
      display:block;
      cursor:pointer;
      top:50%;
      right:0;
      margin-top:-25px;
    }
    #detail_contents{
      padding:10px;
    }
    #detail_text{
      font-family:Helvetica, sans-serif;
      padding:2px;
      font-size:18px;
    }
    #foot_text{
      padding:2px;
      font-family:Helvetica, sans-serif;
      letter-spacing:0;
      font-size:15px;
      position:relative;
      color:#eeeeee;
    }
    @media (max-width: 768px){
      .row{
        flex-wrap:wrap;
      }
    }
    '''

    return css_text

def make_html(output_path, gst, get, coefficients_trend_stride, filter_type, freq1, freq2, main_channel, mic_alpha, mic_c, sample_rate, MIC_maxvalues, PCC_maxvalues, Kendall_maxvalues, sorted_MIC_maxvalues):
    head = html_head()
    configuration = html_configuration(gst, get, coefficients_trend_stride, filter_type, freq1, freq2, main_channel, mic_alpha, mic_c, sample_rate)
    summary = html_summary(gst, get, coefficients_trend_stride, main_channel, MIC_maxvalues, PCC_maxvalues, Kendall_maxvalues, sorted_MIC_maxvalues)
    details = html_details(output_path, gst, get, coefficients_trend_stride, main_channel, MIC_maxvalues, PCC_maxvalues, Kendall_maxvalues, sorted_MIC_maxvalues)
    foot = html_foot()
    script = html_script()
    
    html = head + configuration + summary + details + foot + script

    css = css_text()

    with open(output_path + 'index.html', 'w') as html_file:
        html_file.write(html)

    if not os.path.exists(output_path+'css'):
        os.makedirs(output_path+'css')

    with open(output_path + 'css/style.css', 'w') as css_file:
        css_file.write(css)

    print('Saved HTML file')

