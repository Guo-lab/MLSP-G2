# MLSP-G2
# Blind Source Separation

## Intorduction
The neuromuscular system, driven by Motor Units (MUs), facilitates movement through nerve-tomuscle electrical signals. High-Density Electromyography (HDEMG) is used to track these signals,
but accurately identifying and quantifying active MUs remains challenging.
This project contains Python scripts for processing and analyzing electrophysiological signals. Key features include signal filtering, peak detection, principal component analysis (PCA), K-means clustering, and dynamic time warping (DTW) analysis.

## Dataset
We employ an HDEMG dataset for neuron spike detection and motor unit potential measurement, containing three \texttt{.mat} files: \texttt{steadyforce1.mat}, \texttt{increasingforce1.mat}, and \texttt{increasingforce2.mat}, each representing different muscle force scenarios. The dataset yields a 64-channel time series (\texttt{Out\_mat}) and electrode spatial coordinates (\texttt{Grid\_crds}). Preprocessing involved noise reduction via \texttt{filt\_GRID}.

## Method
Based on biological principles, we recognized that different Motor Units (MUs) possess unique peak characteristics. Therefore, the key to identifying different MUs lies in the detection and analysis of these peaks. 
(Special thanks to TA Prakarsh for providing amazing instructions and hints!)

## Process
![steps](./design/process_with_gray_bg.jpg)

## Result
Optimization could be done in the future step.
<!-- ![Steady](./steady_output.png) -->
<div style="display: inline-block; width: 30%; text-align: center;">
    E.g. Increasing I Force Spikes:
</div>
<br>

<img src="./result/increasing_1_output.png" alt="Steady" width="290" height="1200">


## Installation Instructions
Ensure that Python 3.8 is installed.

