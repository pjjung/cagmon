     ,-----.  ,---.   ,----.   ,--.   ,--.                            ,--.             ,--.        
    '  .--./ /  O  \ '  .-./   |   `.'   | ,---. ,--,--,      ,---. ,-'  '-.,--.,--. ,-|  | ,---.  
    |  |    |  .-.  ||  | .---.|  |'.'|  || .-. ||      \    | .-. :'-.  .-'|  ||  |' .-. || .-. : 
    '  '--'\|  | |  |'  '--'  ||  |   |  |' '-' '|  ||  |    \   --.  |  |  '  ''  '\ `-' |\   --. 
     `-----'`--' `--' `------' `--'   `--' `---' `--''--'     `----'  `--'   `----'  `---'  `----' 
     
The CAGMon is the tool that evaluates the dependence between the primary and auxiliary channels of Gravitational-Wave detectors.   

The goal of this project is to find a systematic way of identifying the abnormal glitches in the gravitational-wave data using various methods of correlation analysis. Usually, the community such as LIGO, Virgo, and KAGRA uses a conventional way of finding glitches in auxiliary channels of the detector - Klein-Welle, Omicron, Ordered Veto Lists, etc. However, some different ways can be possible to find and monitor them in a (quasi-) realtime. Also, the method can point out which channel is responsible for the found glitch. In this project, we study its possible to apply three different correlation methods - maximal information coefficient, Pearson's correlation coefficient, and Kendall's tau coefficient - in the gravitational wave data from the KAGRA detector.

## Status

[![Build Status](https://img.shields.io/badge/version-0.8.5-blue)](https://img.shields.io/badge/version-0.8.5-blue)

[![Build Status](https://img.shields.io/badge/license-%20GPLv3-green)](http://www.gnu.org/licenses/)
[![Build Status](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-brightgreen)](https://minepy.readthedocs.io/en/latest/)
[![Build Status](https://img.shields.io/badge/package-gwpy-yellow)](https://gwpy.github.io)
[![Build Status](https://img.shields.io/badge/package-minepy-yellowgreen)](https://minepy.readthedocs.io/en/latest/)

## References

The CAGMon algorithm is described in
- Piljong Jung, Sang Hoon Oh, Young-Min Kim, Edwin J. Son, John J. Oh, *Optimizing Parameters of Information-Theoretic Correlation Measurement for Multi-Channel Time-Series Datasets in Gravitational Wave Detectors*, [arXiv:2107.03516](https://arxiv.org/abs/2107.03516)
- Piljong Jung, Sang Hoon Oh, Young-Min Kim, Edwin J. Son, John J. Oh, *Identifying and diagnosing coherent associations and causalities between multi-channels of the gravitational wave detector*, [JGW-P2113130](https://gwdoc.icrr.u-tokyo.ac.jp/cgi-bin/private/DocDB/ShowDocument?docid=13130)


## Installation

```sh
$ git clone https://github.com/pjjung/cagmon.git
$ cd cagmon
$ python setup.py install
```

## Syntax of configuration files (.ini)
* Example of full configurations
<pre>
<code>
[GENERAL]
gps_start_time = 1234500000
gps_end_time = 1234599968
stride = 512

[PREPROSECCING]
datasize = 8192
filter_type = highpass (or low/bandpass)
frequency1 = 10 (if bandpass file is applied, two frequency conditions are required; frequncy1 and crequncy2)

[SEGMENT]
defined_condition = LSC_LOCK_STATE_CHANNEL == 10 (or segment_file_path = /path/to/segment/file/)

[CHANNELS]
main_channel = GW-STRAIN_CHANNEL
aux_channels_file_path = /path/to/channel/list/file

[INPUT AND OUTPUT PATHS]
frame_files_path = /path/to/frame/file/folder
output_path = /path/to/output/folder
</code>
</pre> 

* Example of essential configurations
<pre>
<code>
[GENERAL]
gps_start_time = 1234500000
gps_end_time = 1234599968
stride = 512

[SEGMENT]
defined_condition = LSC_LOCK_STATE_CHANNEL == 10 (or segment_file_path = /path/to/segment/file/)

[CHANNELS]
main_channel = GW-STRAIN_CHANNEL
aux_channels_file_path = /path/to/channel/list/file

[INPUT AND OUTPUT PATHS]
frame_files_path = /path/to/frame/file/folder
output_path = /path/to/output/folder
</code>
</pre> 

## Syntax of Channel list files
* Type 1
<pre>
<code>
K1:AUX_CHANNEL_NAME_1
K1:AUX_CHANNEL_NAME_2
K1:AUX_CHANNEL_NAME_3
.
.
.
</code>
</pre> 

* Type 1
<pre>
<code>
K1:AUX_CHANNEL_NAME_1 SAMPLE_RATE
K1:AUX_CHANNEL_NAME_2 SAMPLE_RATE
K1:AUX_CHANNEL_NAME_3 SAMPLE_RATE
.
.
.
</code>
</pre> 

## Execute the CAGMon etude
```sh
$ cagmon --config cagmon_config.ini
```

## License

The CAGMon is following the GNU General Public License version 3. Under this term, you can redistribute and/or modify it.
See [the GNU free software license](http://www.gnu.org/licenses/) for more details.
