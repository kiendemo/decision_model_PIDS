# Datasets (rendered from configs/datasets.yaml)

The 5 in-scope hosts. Day lists are frozen (usable clean days from the transformed-graph byte mass); GT files are listed in config order (attack groups follow it).

## CADETS_E3

- os freebsd; database cadets_e3
- test days: 2018-04-06, 2018-04-11, 2018-04-12, 2018-04-13 (configured: 2018-04-06, 2018-04-11, 2018-04-12, 2018-04-13)
- clean days: 2018-04-02, 2018-04-03, 2018-04-04, 2018-04-05, 2018-04-07, 2018-04-08, 2018-04-09, 2018-04-10; pre-deployment: 2018-04-02, 2018-04-03, 2018-04-04, 2018-04-05; inner day 2018-04-05; k2 reference 2018-04-02, 2018-04-03
- main configuration k = 2; k0 nulls: yes; E1 levels k0: L0, L1, L3; k2: L0, L1, L2, L3, L4; kall: L0, L1, L2, L3, L4
- GT files: E3-CADETS/node_Nginx_Backdoor_06.csv, E3-CADETS/node_Nginx_Backdoor_12.csv, E3-CADETS/node_Nginx_Backdoor_13.csv
- test-day nodes 281,585; evaluation universe 281,589; GT 72 (68 with a test-day card); card days 12

## THEIA_E3

- os linux; database theia_e3
- test days: 2018-04-10, 2018-04-12, 2018-04-13 (configured: 2018-04-10, 2018-04-12, 2018-04-13)
- clean days: 2018-04-03, 2018-04-04, 2018-04-05, 2018-04-09; pre-deployment: 2018-04-03, 2018-04-04, 2018-04-05, 2018-04-09; inner day 2018-04-09; k2 reference 2018-04-03, 2018-04-04
- main configuration k = all; k0 nulls: yes; E1 levels k0: L0, L1, L3; k2: L0, L1, L2, L3, L4; kall: L0, L1, L2, L3, L4
- GT files: E3-THEIA/node_Browser_Extension_Drakon_Dropper.csv, E3-THEIA/node_Firefox_Backdoor_Drakon_In_Memory.csv
- test-day nodes 700,902; evaluation universe 700,902; GT 118 (118 with a test-day card); card days 11

## CLEARSCOPE_E3

- os android; database clearscope_e3
- test days: 2018-04-11, 2018-04-12 (configured: 2018-04-11, 2018-04-12)
- clean days: 2018-04-03, 2018-04-04, 2018-04-05, 2018-04-07, 2018-04-08, 2018-04-09, 2018-04-10; pre-deployment: 2018-04-03, 2018-04-04, 2018-04-05, 2018-04-07, 2018-04-08, 2018-04-09, 2018-04-10; inner day 2018-04-10; k2 reference 2018-04-03, 2018-04-04
- main configuration k = all; k0 nulls: no; E1 levels k0: L3; k2: L2, L4; kall: L2, L4
- GT files: E3-CLEARSCOPE/node_clearscope_e3_firefox_0411.csv
- test-day nodes 111,403; evaluation universe 111,403; GT 41 (41 with a test-day card); card days 10

## optc_h501

- os windows; database optc_h501
- test days: 2019-09-23, 2019-09-24, 2019-09-25 (configured: 2019-09-23, 2019-09-24, 2019-09-25)
- clean days: 2019-09-19, 2019-09-20, 2019-09-21, 2019-09-22; pre-deployment: 2019-09-19, 2019-09-20, 2019-09-21, 2019-09-22; inner day 2019-09-22; k2 reference 2019-09-19, 2019-09-20
- main configuration k = all; k0 nulls: yes; E1 levels k0: L0, L1, L3; k2: L0, L1, L2, L3, L4; kall: L0, L1, L2, L3, L4
- GT files: h501/node_h501_0924.csv
- test-day nodes 1,489,198; evaluation universe 1,489,198; GT 749 (749 with a test-day card); card days 7

## ATLASV2_EDR

- os windows; database atlasv2_edr
- test days: 2022-07-19 (configured: 2022-07-19, 2022-07-20)
- clean days: 2022-07-16, 2022-07-17, 2022-07-18; pre-deployment: 2022-07-16, 2022-07-17, 2022-07-18; inner day 2022-07-18; k2 reference 2022-07-16, 2022-07-17
- main configuration k = all; k0 nulls: yes; E1 levels k0: L0, L1, L3; k2: L0, L1, L2, L3, L4; kall: L0, L1, L2, L3, L4
- GT files: atlasv2_edr/atlasv2_edr_s1.csv, atlasv2_edr/atlasv2_edr_s2.csv, atlasv2_edr/atlasv2_edr_s3.csv, atlasv2_edr/atlasv2_edr_s4.csv, atlasv2_edr/atlasv2_edr_m1.csv, atlasv2_edr/atlasv2_edr_m2.csv, atlasv2_edr/atlasv2_edr_m3.csv, atlasv2_edr/atlasv2_edr_m4.csv, atlasv2_edr/atlasv2_edr_m5.csv, atlasv2_edr/atlasv2_edr_m6.csv
- test-day nodes 152,763; evaluation universe 152,763; GT 48 (48 with a test-day card); card days 5
