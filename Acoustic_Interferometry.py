#############################################################################
#
#    ACOUSTIC INTERFEROMETRY: Read WAV recordings and analyze them to 
#    determine the sound speed using the cross-correlation method and geometric
#    parameters
#
#    Copyright (C) 2024  Daniel Pardo & Iván Martí (UV), Spain
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#############################################################################


# LOAD NEEDED LIBRARIES:
import pylab as pl
import numpy as np
import scipy as sp
import struct
import wave
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
import os

from scipy import optimize as spopt
from scipy.io import wavfile
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

#############################################################################
## SCRIPT CONFIGURATION: ##
#############################################################################

## SIGNALS THAT HAVE BEEN RECORDED:
RECEIVERS = []
ORIGINAL_AUDIO = [] 

############################################
### AUDIO CHARACTERISTICS:
SAMPLING_RATE = 44100  ## (i.e., CD quality)

################
#####
SAMPLE_WIDTH = 16  ## (i.e., CD quality)
DATA_TYPE = 'h' ## (i.e., CD quality; short integer)
#####

####### OTHER OPTIONS: 
#####
#SAMPLE_WIDTH = 32 
#DATA_TYPE = 'i' 
#####
#SAMPLE_WIDTH = 64 
#DATA_TYPE = 'q' 
#######
#####################

DURATION = 10.0  ## (in seconds)
VOLUME = 0.9  ## (w.r.t. maximum. BEWARE not to saturate (VOLUME >= 1.0))

## SOURCE 0:
NU0 = 10000.0  # (center frequency in Hz)
DNU0 = 500.0 # (bandwidth in Hz)

## SOURCE 1:
NU1 = 5000.0  # (center frequency in Hz)
DNU1 =  500.0 # (bandwidth in Hz)

# NUMBER OF SAMPLES IN AUDIO FILE:
NSAMPLES = int(SAMPLING_RATE*DURATION)

# WIDTH OF FREQUENCY CHANNELS (FOR FFT):
DF = float(SAMPLING_RATE)/float(NSAMPLES)

############################################
### EXPERIMENT DIMENSIONS:
a =  # (in metres, length separation between speakers)
dX =  # (in metres, offset of first speaker w.r.t. zero-position receiver)
D =  # (in metres, length between speakers and micropohones' line)
K = # (in metres, constant)

############################################
### VARIABLES TO CARRY OUT DATA PROCESSING:
SAMP_INI = 22050*1  ## First sample to use in the cross-correlations
# (e.g. an audio of 44 100 samples we recommend a value of 22 050)

SAMP_DUR = 8 ## Time integration of the cross-correlations (in seconds)
# (e.g. an audio of 10 s of duration we recommend 8 s of integration)

NCHAN = 256

ZPAD = 8 ## This parametre increase the signal with zeros ('Zeropadding')

plotFringes = True ## Plot the fringes for ALL baselines?

############################################
## POSITION OF RECORDS
NAME = [] ## Name of each micro's position
LOCATIONS = [] # in m

## NAME OF RECORDINGS THAT DON'T WANT TO INCLUDE:
BADS = []

## DELAYS BETWEEN CHANNELS (You have to enter it manually)
INI_DELAYS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 

### END OF CONFIGURATION.   
############################################################################# 

#############################################################################
## FUNCTIONS: ##
#############################################################################

############################################
## GENERATE SIGNALS:
def generate():
    
    FILENAME = 'DOUBLE_SOURCE'
    
    ## PREPARE WAVE FILE:
    NBYTES = int(SAMPLE_WIDTH/8)
    OUTPUT = wave.open('%s.wav'%FILENAME,'w')
    OUTPUT.setparams((2,NBYTES,int(SAMPLING_RATE),NSAMPLES,'NONE','not compressed'))
    
    #############
    ## (I) GENERATE WHITE NOISES (i.e., Gaussian):

    # Channel 0 has source 0:
    NOISE_0 = np.random.normal(0.,1.,NSAMPLES)

    # Channel 1 has source 1:
    NOISE_1 = np.random.normal(0.,1.,NSAMPLES)
    #############

    #############
    ## (II) APPLY DIGITAL FILTERS:

    # Convert to frequency space using FFT:
    FFT_SIGNAL_0 = np.fft.fft(NOISE_0)
    FFT_SIGNAL_1 = np.fft.fft(NOISE_1)

    ## Filter signal 0 (i.e., make zeros out from [numin, numax]):
    NUMIN = NU0-DNU0/2.
    NUMAX = NU0+DNU0/2.
    CHANMIN = int(NUMIN/DF)
    CHANMAX = int(NUMAX/DF)
    print('CHARACTERISTICS SOURCE_0:\n')
    print('WIDTH OF FREQUENCY CHANNEL:', DF,'\nSOURCE_0 EXTENDS FROM THE FREQUENCY:', NUMIN, 'Hz TO', NUMAX, 'Hz')
    print('\nNUMBER OF SAMPLES:', NSAMPLES, '\nSOURCE_0 BETWEEN SAMPLES',CHANMIN,'AND', CHANMAX,'\n')
    FFT_SIGNAL_0[:CHANMIN] = 0.0
    FFT_SIGNAL_0[-CHANMIN:] = 0.0
    FFT_SIGNAL_0[CHANMAX:NSAMPLES-CHANMAX] = 0.0
    SIGNAL_0 = np.real(np.fft.ifft(FFT_SIGNAL_0))

    ## Filter signal 1 (i.e., make zeros out from [numin, numax]):
    NUMIN = NU1-DNU1/2.
    NUMAX = NU1+DNU1/2.
    CHANMIN = int(NUMIN/DF)
    CHANMAX = int(NUMAX/DF)
    print('CHARACTERISTICS SOURCE_1:\n')
    print('WIDTH OF FREQUENCY CHANNEL:', DF,'\nSOURCE_1 EXTENDS FROM THE FREQUENCY:', NUMIN, 'Hz TO', NUMAX, 'Hz')
    print('\nNUMBER OF SAMPLES:', NSAMPLES, '\nSOURCE_1 BETWEEN SAMPLES',CHANMIN,'AND', CHANMAX,'\n')
    FFT_SIGNAL_1[:CHANMIN] = 0.0
    FFT_SIGNAL_1[-CHANMIN:] = 0.0
    FFT_SIGNAL_1[CHANMAX:NSAMPLES-CHANMAX] = 0.0
    SIGNAL_1 = np.real(np.fft.ifft(FFT_SIGNAL_1))
    #############

    #############
    ## (III) DIGITIZE FILTERED SIGNALS IN AMPLITUDE:

    # Maximum amplitude that can be stored in wave file:
    MAX_DIGI = np.power(2.,SAMPLE_WIDTH-1.)

    # Maximum amplitudes of the filtered signals:
    MAX_AMP_0 = np.max(np.abs(SIGNAL_0))
    MAX_AMP_1 = np.max(np.abs(SIGNAL_1))

    # We will store digitized signals here:
    DIGI_0 = np.zeros(NSAMPLES,dtype=int)
    DIGI_1 = np.zeros(NSAMPLES,dtype=int)

    # Store signals, properly digitized in amplitude:
    DIGI_0[:] = VOLUME*SIGNAL_0/MAX_AMP_0*MAX_DIGI
    DIGI_1[:] = VOLUME*SIGNAL_1/MAX_AMP_1*MAX_DIGI

    # Plex stereo signal (sample by sample):
    DIGI_STEREO = np.zeros(2*NSAMPLES,dtype=int)
    DIGI_STEREO[::2] = DIGI_0 # Right channel
    DIGI_STEREO[1::2] = DIGI_1 # Left channel:
    #############

    #############
    ## (IV) SAVE WAVEFILE:

    # Pack signal in proper byte format:
    FINAL_SOURCE = struct.pack(DATA_TYPE*NSAMPLES*2,*DIGI_STEREO)

    # Write and close:
    OUTPUT.writeframes(FINAL_SOURCE)
    OUTPUT.close()
    
    print('The audio has been created with the name:', FILENAME, '.wav')
  
############################################    
## GOLOMB'S RULES:
def golomb(number):
    number = np.array(number, 'int')
    global a,D
    #############
    ## This function can estimated the position of microphones from the 
    ## velocity's precision you want to get in your experiment. Moreover,
    ## the function checks if its possible reprouce the result with your 
    ## experiment dimensions.
    #############
    
    #############
    ## (I) DEFINITION OF VARIABLES
    ErrC = 2.0     # in m/s.
    ErrTau = 1./SAMPLING_RATE  # in s.
    
    C = 340 # in m/s
    
    amin = 0.4 ; amax = a; na = 1000
    Dmin = 0.5; Dmax = D; nD = 1000

    aa = np.linspace(amin,amax,na)
    DD = np.linspace(Dmin,Dmax,nD)
    #############
    
    #############
    ## (II) SELECTION OF NUMBER OF MICROPHONE'S POSITIONS
    if number < 4:
        print('You have to introduce a minimum of 4 position for run the program')
        sys.exit()
    elif number == 4:
        GOLI = [0,1,4,6]  # Golomb rule of order 4 
    elif number == 5:
        GOLI = [0,1,4,9,11]  # Golomb rule of order 5
    elif number == 6:   
        GOLI = [0,1,4,10,12,17] # Golomb rule of order 6
    elif number == 7:
        GOLI = [0,1,4,10,18,23,25]  # Golomb rule of order 7
    elif number == 8:
        GOLI = [0,1,4,9,15,22,32,34] # Golomb rule of order 8
    elif number == 9:
        GOLI = [0,1,5,12,25,27,35,41,44] # Golomb rule of order 9
    elif number == 10:
        GOLI = [0,1,6,10,23,26,34,41,53,55]  # Golomb rule of order 10
    elif number == 11:
        GOLI = [0,1,4,13,28,33,47,54,64,70,72]  # Golomb rule of order 11
    elif number == 12:
        GOLI = [0,2,6,24,29,40,43,55,68,75,76,85] # Golomb rule of order 12
    elif number == 13:
        GOLI = [0,2,5,25,37,43,59,70,85,89,98,99,106]  # Golomb rule of order 13
    elif number == 14:
        GOLI = [0,4,6,20,35,52,59,77,78,86,89,99,122,127] # Golomb rule of order 14
    elif number == 15:
        GOLI = [0,4,20,30,57,59,62,76,100,111,123,136,144,145,151]  # Golomb rule of order 15
    elif number == 16:
        GOLI = [0,1,4,11,26,32,56,68,76,115,117,134,150,163,168,177] # Golomb rule of order 16
    elif number == 17:
        GOLI = [0,5,7,17,52,56,67,80,81,100,122,138,159,165,168,191,199] # Golomb rule of order 17
    elif number == 18:
        GOLI = [0,2,10,22,53,56,82,83,89,98,130,148,153,167,188,192,205,216] # Golomb rule of order 18
    elif number == 19:
        GOLI = [0,1,6,25,32,72,100,108,120,130,153,169,187,190,204,231,233,242,246]  # Golomb rule of order 19
    elif number == 20:
        GOLI = [0,1,8,11,68,77,94,116,121,156,158,179,194,208,212,228,240,253,259,283] # Golomb rule of order 20
    else:
        print("This number of position is not available for this code")
        sys.exit()
        
    Golomb = np.array(GOLI, dtype=float)
    GOLI = np.copy(Golomb)
    GOLI /= np.max(GOLI)   # Standardization of Golomb's rules
    GOLI = K * GOLI
    #############
     
    #############
    ## (III) CALCULATE THEORETICAL DISTANCE BETWEEN SPEAKERS AND MICROS
    RESUL = np.zeros((na,nD))
    
    for ii,ai in enumerate(aa):
      for jj,Di in enumerate(DD):
        
        a = ai # en m (length between speakers)
        D = Di  # en m (distance between speaker's ceneter and Golomb rule)
        
        BASEL = []
        
        for i in range(number):
            for j in range(i+1,number):
                r1i = np.sqrt(D**2. + (GOLI[i])**2.)       # Speaker 1 to micro i
                r2i = np.sqrt(D**2. + (a-GOLI[i])**2.)  # Speaker 2 to micro i
                r1j = np.sqrt(D**2. + (GOLI[j])**2.)      # Speaker 1 to micro j
                r2j = np.sqrt(D**2. + (a-GOLI[j])**2.) # Speaker 2 to micro j
                dif = (r2i-r2j) - (r1i-r1j)      # Differences between distances
                BASEL.append(dif) 
           
        BASEL = np.array(BASEL)
        AVER = np.average(BASEL) # Average
        SQAVER = np.average(np.power(BASEL,2)) # Quadratic average

        DIFF = (SQAVER - AVER**2.)*len(BASEL) 

        sqrtK = ErrTau*C**2./np.sqrt(DIFF)/ErrC # Based on linear regression
        RESUL[ii,jj] = sqrtK**2.  
    #############
    
    #############
    ## (IV) PARAMETER ESTIMATION PLOT
    fig = pl.figure()
    sub = fig.add_subplot(111)
    im = sub.imshow(RESUL, origin='lower', extent=[Dmin,Dmax,amin,amax], aspect=(1/((amax-amin)/(Dmax-Dmin))))
    sub.set_xlabel('D (m)', fontsize = 12)
    sub.set_ylabel('a (m)', fontsize = 12)
    cb = pl.colorbar(im)
    cb.set_label(r'$\alpha$ (m)',fontsize = 12)
    
    if plotFringes:
        filename = 'Golomb_estimation.png'
        pl.savefig(filename)
        pl.show()
    else:
        pl.show()
    
############################################    
## PLOT THE SPECTRUM OF ORIGINAL AUDIO:
def original_audio():
    #############
    ## This function show the spectrum of original audio in frequency domain
    #############
    
    AUDIO = []
    for name in ORIGINAL_AUDIO:
        sample_rate, wave = wavfile.read(name, 'r')
        AUDIO.append(wave[:].astype(np.float32))
        
    DATA = []
    # Checking the number of channels
    if AUDIO[0].ndim > 1:         ## Stereo audios
        if AUDIO[0].ndim == 2:
            LEFT = AUDIO[0][:,0]
            RIGHT = AUDIO[0][:,1]
            DATA.append(0.5*LEFT + 0.5*RIGHT)
           
    elif AUDIO[0].ndim == 1:      ## Mono audios
            DATA.append(AUDIO[0])
            
    # Compute the FFT and get its amplitude:
    FFT = np.fft.fft(DATA[0])    
    AMP = np.abs(FFT)

    # Maximum frequency of the FFT (i.e., Nyquist):
    NUMAX = sample_rate//2
    NFREQ = len(DATA[0])//2
    
    # List of frequency channels of the FFT:
    FREQS = np.linspace(0.,NUMAX,NFREQ)
    title = 'FFT in Original Audio'
    
    # Plot! (scale frequencies in kHz):
    pl.plot(FREQS/1.e3, AMP[:NFREQ]/np.max(AMP),'-b') 
    pl.title(title, loc = "center", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
    pl.xlabel('Frequency (kHz)')
    pl.ylabel('Amp. (Norm)')
    
    if plotFringes:
        filename = 'FFT_Original_Audio.png'
        pl.savefig(filename)
        pl.show()
    else:
        pl.show()
        
############################################    
## PLOT THE SPECTRUM OF DIFFERENT RECORDINGS:
def recordings_spectrum(number):
    number = np.array(number, 'int')
#############
## This function show the spectrum of all recordings in frequency domain
#############
    for rec in RECEIVERS:
      sample_rate, wave = wavfile.read(rec,'r')
      SIGNALS.append(wave[:].astype(np.float32))
      
    DATES = []
    for i in range(number):
    # Checking the number of channels
        if SIGNALS[i].ndim > 1:         ## Stereo audios
            if SIGNALS[i].ndim == 2:
                LEFT = SIGNALS[i][:,0]
                RIGHT = SIGNALS[i][:,1]
                DATES.append(0.5*LEFT + 0.5*RIGHT)
        elif SIGNALS[i].ndim == 1:      ## Mono audios
                DATES.append(SIGNALS[i])    
                
    for i in range(number):
        # Compute the FFT and get its amplitude:
        FFT = np.fft.fft(DATES[i])    ##(Transformada de Fourier)
        AMP = np.abs(FFT)

        # Maximum frequency of the FFT (i.e., Nyquist):
        NUMAX = sample_rate//2
        NFREQ = len(DATES[i])//2

        # List of frequency channels of the FFT:
        FREQS = np.linspace(0.,NUMAX, NFREQ)

        # Configuration of the microphone's position
        filename = str(NAME[i])
        title = 'FFT in ' + filename + ' position'
        
        # Plot! (scale frequencies in kHz):
        pl.plot(FREQS/1.e3, AMP[:NFREQ]/np.max(AMP),'-b') ## (Valores x, Valores y, Diseño gráfico)
        pl.title(title, loc = "center", fontdict = {'fontsize':14, 'fontweight':'bold', 'color':'tab:blue'})
        pl.xlabel('Frequency (kHz)')
        pl.ylabel('Amp. (Norm)')
        
        if plotFringes:
            filename = 'Posicion_' + filename + '.png'
            pl.savefig(filename)
            pl.show()
        else:
            pl.show()

###########################################    
## PLOT EACH MICROPHONE'S DATES:
def bad_position(results, fit_dates):
#############
## This function show fit's figure with each microphone position's dates
#############
    
    # Variable for saving each microphone dates
    DATES = []
    POS = []
    
    for i in range(len(LOCATIONS)):
            if not RECEIVERS[i] in BADS:
                POS.append(NAME[i])
    POS = np.array(POS)
    
    # Read and clasificate dates
    for k in POS:
        STORE = [] # In this variable, it will introduce tha dates of the same position
        l = 0 # Position in the variable which results are stored
        for i in range(len(POS)):
            for j in range(i+1,len(POS)):
                if (POS[i] == k) or (POS[j] == k):
                    STORE.append(results[l])
                    l = l+1
                else:
                    l = l+1
        DATES.append(STORE)
    DATES = np.array(DATES)    
    
    for i in range(len(POS)):
        
        # Plot measurements
        fig2 = pl.figure()
        sub2 = fig2.add_subplot(111)
        sub2.plot(DATES[i][:,0],1000.*DATES[i][:,1],'ok')
        sub2.errorbar(DATES[i][:,0],1000.*DATES[i][:,1],1000.*DATES[i][:,2],fmt='k',linestyle='none')

        sub2.set_xlabel('Diff. camino (m)')
        sub2.set_ylabel('Diff. retraso (ms)')

        # Plot the best-fit model:
        PLDELS = np.linspace(np.min(results[:,0]),np.max(results[:,0]),100)
        sub2.plot(PLDELS,1000.*soundSpeed(fit_dates[0],dels=PLDELS),'-b',lw=2)

        if plotFringes:
            filename = ('Fit_with_microphone_%i.png'%POS[i])
            pl.savefig(filename)
            pl.show()
        else:
            pl.show()
    print('The program has showed each position results.')
    print('If you showed a set that doesn\'t follow the linear trend, include the audio in BADS variable and run the code again or repeat the measure')

############################################    
## MONTECARLO'S METHOD:
def montecarlo(results,N,a,D,dX, n_bins, variable_fix = [False,False,False,False]):
#############
## This function allow study with accuracity the geometry  of the experiment
#############
## You have to introduce all geometry variables, the number of the distribution
## and in variable_fix introduce true if you want analyze the more contribution
    if variable_fix[0]:
        a_m = [a for i in range(N)]
    else:
        a_m = np.random.normal(a, 0.005, N) 
     
    if variable_fix[1]:
        D_m = [D for i in range(N)]
    else:
        D_m = np.random.normal(D, 0.005, N)

    if variable_fix[2]:
        dX_m = [dX for i in range(N)]
    else:
        dX_m = np.random.normal(dX, 0.001, N)
    
    DIST = [] # Variable position
    SPEED = [] # Variable speed of sound
    
    POS = []
    
    for i in range(len(LOCATIONS)):
           if not RECEIVERS[i] in BADS:
               POS.append(LOCATIONS[i])
    POS = np.array(POS)
       
    if variable_fix[3]:
       POS_m = [[POS[k] for i in range(N)] for k in range(len(POS))]
    else:
       POS_m = [np.random.normal(i, 0.001, N) for i in POS]
       
    def soundSpeed(p):
       return (results[:,0]/p[0]+p[1] - results[:,1])
       
    for i in range(N):
               for k in range(len(POS)):
                   for j in range(k+1, len(POS)):
                       Dist_0i = np.sqrt((POS_m[k][i]-dX_m[i])**2. + D_m[i]**2.)
                       Dist_1i = np.sqrt((a_m[i] - (POS_m[k][i]-dX_m[i]))**2. + D_m[i]**2.)

                       Dist_0j = np.sqrt((POS_m[j][i]-dX_m[i])**2. + D_m[i]**2.)
                       Dist_1j = np.sqrt((a_m[i] - (POS_m[j][i]-dX_m[i]))**2. + D_m[i]**2.)
                       DiffDist = (Dist_1i-Dist_1j) - (Dist_0i-Dist_0j)
             
                       DIST.append(DiffDist)
               results[:,0] = DIST
               DIST = []
       
               c = spopt.leastsq(soundSpeed,[1.,0.],full_output=True)
               SPEED.append(c[0][0])
       
    SPEED = np.array(SPEED)
    print('standard desviation = ', np.std(SPEED))
    # Plot speed histogram
    fig, ax = pl.subplots(tight_layout=True)
    ax.hist(SPEED, bins = n_bins)
   
    if plotFringes:
       filename = 'Montecarlo_Histogram.png'
       pl.savefig(filename)
       pl.show()
    else:
       pl.show()
       
#############################################################################
## SCRIPT STARTS: ##
#############################################################################

## CHECKING VARIABLES:
    
# Firstly, check if there is audio for doing the experiment
if ORIGINAL_AUDIO == []:
    print('You have to insert the original audio or generate one with the function "generate()"')
    sys.exit()
else:
    pass 

# Then, check if there are different locations of microphone for each recording
if LOCATIONS == []:
    print('You have to insert microphone\'s position')
    print('In this script, we propose using the Golomb\'s rules with the function "golomb(number of position)"')
    sys.exit()
else:
    pass

# We last need to check if there are recordings
if RECEIVERS == []:
    print('You have to insert recordings of each given position in the same order as positions')
    quit()
elif len(LOCATIONS) == len(RECEIVERS):
    SIGNALS = []
    for rec in RECEIVERS:
      sample_rate, wave = wavfile.read(rec,'r')
      SIGNALS.append(wave[:].astype(np.float32))
    pass
else: 
    print('The number of position doesn\'t match the number of recordings')
    sys.exit()
    
## DEFINITION OF IMPORTANT VARIABLES FOR THE STUDY OF TEMPORAL DATA:
    
DT = 1./sample_rate # Time between samples (i.e., time lag))
NSAMP = int(sample_rate*SAMP_DUR) # Number of sample in total
SAMP_FIN = SAMP_INI + NSAMP # Last sample to take
DF = float(sample_rate)/float(NSAMP) # Frequency resolution of the FFT

print('All variables are correct to run the program. Let\'s go with it!!\n')

############################################
#############
## (I) SEPARATE THE SIGNALS OF EACH SPEAKER WITH A DIGITAL FILTER:
##      (a) Do FFT and zero the amplitudes of the undesired frequencies.
##      (b) Come back to time domain (e.g, do IFFT)

## (a) Do FFT and zero the amplitudes of the undesired frequencies.

# First and last FFT channels of the first signal:
CHANMIN_0 =  int((NU0 - DNU0/2.)/DF)
CHANMAX_0 =  int((NU0 + DNU0/2.)/DF)
NBW_0 = CHANMAX_0-CHANMIN_0

# First and last FFT channels of the last signal:
CHANMIN_1 =  int((NU1 - DNU1/2.)/DF)
CHANMAX_1 =  int((NU1 + NU1/2.)/DF)
NBW_1 = CHANMAX_1-CHANMIN_1

# Show digital filter for each signal:
print('SPEC 0 from %i to %i and from %i to %i'%(CHANMIN_0,CHANMAX_0,NSAMP-CHANMAX_0,NSAMP-CHANMIN_0))
print('SPEC 1 from %i to %i and from %i to %i'%(CHANMIN_1,CHANMAX_1,NSAMP-CHANMAX_1,NSAMP-CHANMIN_1))

# Save the filtered signals in these lists:
BB_SIGNALS_0 = []
BB_SIGNALS_1 = []

# Just an auxiliary variable:
BB_FFT = np.zeros(NSAMP,dtype=np.complex64)

## FILTER THE SIGNALS FOR ALL MICROPHONE POSITIONS:
for i in range(len(LOCATIONS)):

# Get the signal (from first to last usable sample):
  Si = np.copy(SIGNALS[i][SAMP_INI-INI_DELAYS[i]:SAMP_FIN-INI_DELAYS[i],0])

# Compute the FFT of the whole stream:
  FFT_i = np.fft.fft(Si)
  
## (b) Come back to time domine (e.g, do IFFT)

# Use only the frequency channels of the first speaker signal:
  BB_FFT[:] = 0.0
  BB_FFT[CHANMIN_0:CHANMAX_0] = FFT_i[CHANMIN_0:CHANMAX_0] # Real part
  BB_FFT[NSAMP-CHANMAX_0:NSAMP-CHANMIN_0] = FFT_i[NSAMP-CHANMAX_0:NSAMP-CHANMIN_0] # Imaginary part

# Come back to time domain in the first speaker:
  BB_SIGNALS_0.append(np.fft.ifft(BB_FFT).real)

# Repeat the same procedure for the signal of the second speaker:
  BB_FFT[:] = 0.0
  BB_FFT[CHANMIN_1:CHANMAX_1] = FFT_i[CHANMIN_1:CHANMAX_1]
  BB_FFT[NSAMP-CHANMAX_1:NSAMP-CHANMIN_1] = FFT_i[NSAMP-CHANMAX_1:NSAMP-CHANMIN_1]
  BB_SIGNALS_1.append(np.fft.ifft(BB_FFT).real)
############################


############################################
#############
## (II) LET'S CORRELATE! (WE WILL AVOID LONG-SCALE CLOCK DRIFTS)
##      (a) Prepare variables for save all correlations 
##      (b) Cross-correlate different recordings (with an option of separate bad recordings)
##      (c) Come back to time domain and get the differential time delay

## (a) Prepare variables for save all correlations

# Maximum delay that we can sample with NCHAN lags:
MAXDEL = float(NCHAN)/sample_rate/2.

# Array of delay values:
Delays = np.linspace(-MAXDEL, MAXDEL, ZPAD*NCHAN)

# Number of "correlation chunks" (i.e., accumulations):
NINTEG = NSAMP//NCHAN

print('Will perform %i fringe accumulations' %NINTEG)

# Book some memory:
FRINGE_0 = np.zeros(ZPAD*NCHAN,dtype=np.complex64)
FRINGE_1 = np.zeros(ZPAD*NCHAN,dtype=np.complex64)

Si_0 = np.zeros(NCHAN)
Si_1 = np.zeros(NCHAN)
Sj_0 = np.zeros(NCHAN)
Sj_1 = np.zeros(NCHAN)

DELAY_0 = np.zeros(ZPAD*NCHAN,dtype=np.float32)
DELAY_1 = np.zeros(ZPAD*NCHAN,dtype=np.float32)

# Will save the measured delays here:
RESULTS = []
BASELINES = []
    
# Prepare figures (if plotFringes = True):
if plotFringes:
  figFringe = pl.figure()
  subFringe = figFringe.add_subplot(111)
  os.system('rm -rf FRINGE_*-*.png')
  
## (b) Cross-correlate different recordings (with an option of separate bad recordings)

for i in range(len(LOCATIONS)-1):
    for j in range(i+1,len(LOCATIONS)):

# Is this pair of positions "good"?
        isBad = False
        for k in BADS:
            if (RECEIVERS[i] in k) or (RECEIVERS[j] in k):
                print('Skipping (%i,%i)'%(NAME[i],NAME[j]))
                isBad = True
                break
# If it's good, proceed:
        if not isBad:
            
# Accumulate integrations:
          FRINGE_0[:] = 0.0 ; FRINGE_1[:] = 0.0 
          for ti in range(NINTEG):
# Get the streams in chunks of size = NCHAN:
              Si_0[:] = BB_SIGNALS_0[i][ti*NCHAN:(ti+1)*NCHAN]
              Si_1[:] = BB_SIGNALS_1[i][ti*NCHAN:(ti+1)*NCHAN]
              Sj_0[:] = BB_SIGNALS_0[j][ti*NCHAN:(ti+1)*NCHAN]
              Sj_1[:] = BB_SIGNALS_1[j][ti*NCHAN:(ti+1)*NCHAN]

# Cross-correlate and accumulate (using zero-padding):
              FRINGE_0[(ZPAD*NCHAN-NCHAN)//2:(ZPAD*NCHAN+NCHAN)//2] += np.fft.fft(Si_0)*np.conjugate(np.fft.fft(Sj_0))
              FRINGE_1[(ZPAD*NCHAN-NCHAN)//2:(ZPAD*NCHAN+NCHAN)//2] += np.fft.fft(Si_1)*np.conjugate(np.fft.fft(Sj_1))
        
## ACCUMULATION FINISHED!

## (c) Come back to time domain and get the differential time delay

# Now, go to delay space:
          DELAY_0[:] = np.fft.fftshift(np.abs(np.fft.ifft(np.fft.fftshift(FRINGE_0))))
          DELAY_1[:] = np.fft.fftshift(np.abs(np.fft.ifft(np.fft.fftshift(FRINGE_1))))

# Location of the fringe peaks:
          i0_Peak = np.argmax(DELAY_0)
          i1_Peak = np.argmax(DELAY_1)

# Delay values at the peaks:
          fitDelay0 = Delays[i0_Peak]
          fitDelay1 = Delays[i1_Peak]

# We take the errors as the fringe spacing:
          fitDelay0Err = 1./(2.*NU0)
          fitDelay1Err = 1./(2.*NU1)

# Normalize de "fringes", for plotting:
          DELAY_0 /= np.max(DELAY_0)
          DELAY_1 /= np.max(DELAY_1)

# Measured differential delay (i.e., between both signals):
          DiffDelay = fitDelay0 - fitDelay1
          DiffDelayErr = np.sqrt(fitDelay1Err**2. + fitDelay0Err**2.)
          AvgDelay = (Delays[i0_Peak]+Delays[i1_Peak])*0.5

          print('FOR MICS %i-%i, DELAY_0 is: %.4f ms (%i chans)'%(NAME[i],NAME[j],fitDelay0*1000.,int(fitDelay0*sample_rate)))

############################################
#############
## (III) COMPUTE DELAYS BETWEEN DISTANCES FROM SPEAKERS TO MICROPHONE (AND PLOT)
##     (a) Get the difference between distances
##     (b) If you want plot and save figures 

## (a) Get the difference between distances
          Dist_0i = np.sqrt((LOCATIONS[i]-dX)**2. + D**2.)
          Dist_1i = np.sqrt((a - (LOCATIONS[i]-dX))**2. + D**2.)

          Dist_0j = np.sqrt((LOCATIONS[j]-dX)**2. + D**2.)
          Dist_1j = np.sqrt((a - (LOCATIONS[j]-dX))**2. + D**2.)
          DiffDist = (Dist_1i-Dist_1j) - (Dist_0i-Dist_0j)

## (b) If you want plot and save figures 
          if plotFringes:
              subFringe.cla()
              subFringe.plot(1000.*Delays,DELAY_0,'-r',label='Speaker 0')
              subFringe.plot(1000.*Delays,DELAY_1,'-b',label='Speaker 1')
              subFringe.plot(1000.*fitDelay0,1.0,'xr')
              subFringe.plot(1000.*fitDelay1,1.0,'xb')

              subFringe.set_xlim((1000.*(AvgDelay-MAXDEL),1000.*(AvgDelay+MAXDEL)))
              subFringe.set_xlabel('Delay (ms)')
              subFringe.set_ylabel('X-corr. Amp. (Norm)')
              subFringe.set_title('RECV. %i and %i (dDist = %.4f ; dTau = %.4f ; dTau_Teo = %.4f'%(NAME[i],NAME[j],DiffDist, DiffDelay*1000., DiffDist/340.*1000.))
        
              pl.savefig('FRINGE_%i-%i.png'%(NAME[i],NAME[j]))
        
############################

############################################
#############
## (IV) FIT THE SOUND SPEED WITH A LINEAR REGRESSION
##     (a) Save all the results needed to fit the sound speed
##     (b) Fit the speed of sound
##     (c) Plot measurements
          RESULTS.append([DiffDist,DiffDelay,DiffDelayErr])
          BASELINES.append([LOCATIONS[i],LOCATIONS[j]])

## (a) Save all the results needed to fit the sound speed
RESULTS = np.array(RESULTS)
BASELINES = np.array(BASELINES)

## (b) Fit the speed of sound
## This function returns:
##     - The residuals of a model if "dels" is None (default)
##     - The model predictions for all the differential delays 
##       stored in "dels" ("dels" is assumed to be an array).
def soundSpeed(p, dels=None):
  if dels is None:
    return (RESULTS[:,0]/p[0]+p[1] - RESULTS[:,1])
  else:
    return dels/p[0]+p[1]

def linear_model(fit_dates,x):
    m,n = fit_dates
    return m*x + n

c = spopt.leastsq(soundSpeed,[1.,0.],full_output=True)
cErr = np.sqrt(c[1][0,0]*np.sum(c[2]['fvec']**2.)/(len(RESULTS[:,0])-2.))

print('Sound Speed (m/s): %.2f +/- %.2f '%(c[0][0],cErr))

params = [1/c[0][0], c[0][1]] # Save fit parametres in a variable

## Calculate some stadistics
y_fit = linear_model(params, RESULTS[:,0])
residuals = RESULTS[:,1] - y_fit
ss_total = np.sum((RESULTS[:,1]-np.mean(RESULTS[:,1]))**2)
ss_residual = np.sum(residuals**2)
r_squared = 1 - (ss_residual / ss_total)
std_error = np.sqrt(ss_residual / (len(RESULTS[:,1])-2))

print('Fit stadistics:\nR squared: %.2f\n Standard error of the regression: %.3E' %(r_squared, std_error))

## (c) Plot measurements
fig2 = pl.figure()
sub2 = fig2.add_subplot(111)
sub2.plot(RESULTS[:,0],1000.*RESULTS[:,1],'ok')
sub2.errorbar(RESULTS[:,0],1000.*RESULTS[:,1],1000.*RESULTS[:,2],fmt='k',linestyle='none')

sub2.set_xlabel('Diff. camino (m)')
sub2.set_ylabel('Diff. retraso (ms)')

# Plot the best-fit model:
PLDELS = np.linspace(np.min(RESULTS[:,0]),np.max(RESULTS[:,0]),100)
sub2.plot(PLDELS,1000.*soundSpeed(c[0],dels=PLDELS),'-b',lw=2)

pl.savefig('RESULTS_ALL.png')
pl.show()

print('\nTo improve the result you can compare fit\'s figure with each position with the function "bad_position(results, fit_dates)"')
