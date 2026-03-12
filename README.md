# Lee V1-CA1 project

This repository contains analysis code for the V1-CA1 project. 

## Data

Simultaenous electrophysiological recording in the primary visual cortex (V1) and hippocampus CA1 of freely moving rats performing a spatial alternation task on the W-track. Typically four Livermore polymer probes are implanted in a single rat (bilateral V1 and CA1). The data consists of sleep and run epochs. During sleep epochs, the animal is placed in a sleep box. During run epochs, the animal performs spatial alternation on the W-track to get milk reward. The animal's position is tracked with overhead camera in both epoch types. For some run epochs, visual stimuli are presented in various parts of the W-track maze, including the left and right arms (both sides) and the middle segment that joins the three arms. In at least one run epoch (typically `08_r4`) the animal performs the task in darkness (<10^2 R*/s/rod).

Note that this repo does not contain the data. The data lives on Frank lab servers. It only contains analysis scripts. 

## Main findings



## Organization

The analysis scripts can be found in src/v1ca1. They are organized based on what analysis type. 

## Layout

- `src/v1ca1/`: Python package containing the copied top-level Python source files.
