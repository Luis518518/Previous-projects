#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:37:35 2021

@author: luis
"""

import CWB_ReadLogJobs, CWB_ReadStrain
import numpy as np
import sys
import pylab
import pandas as pd
import math

FolderName        = 'O2_L1H1_RUN2_SIM_SCH'
pc                = 'Luis'      
DataPath          = '/home/luis/Desktop/Delfin/Ondas-Gravitacionales-Delf/'

if   FolderName[8:11] == 'RUN' or FolderName[8:11] == 'SEG' or FolderName[8:14] == 'HYBRID':
    fs                = 2048  
                     
else:
    print('PILAS PERRITO: define sampling frequency for the selected project')
    sys.exit()              
wins=.5
flow              = 10                            
doplot            = 1
doprint           = 1

Dat=[]
DattestH=[]
DatH=np.zeros(shape=(1098,28,50))
DattestL=np.zeros(shape=(366,28,50))
DatL=np.zeros(shape=(1098,28,50))
DattestH=np.zeros(shape=(366,28,50))
col=[]
aux=[]
auxt=[]
name=[]
whiteH={}
arr=[]
# if   FolderName == 'O2_L1H1_RUN2_SIM_SCH':
#JOBS   = np.array([2,3,4,5])

# Define jobs for the selected FolderName 
if   FolderName == 'O2_L1H1_RUN2_SIM_SCH':
    JOBS   = np.array([2,3,4,5])
elif FolderName == 'O2_L1H1_RUN3_SIM_SCH':
    JOBS   = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15])
else:
    print('PILAS PERRITO: Unknown FolderName')
    sys.exit()
    
# Define factors
if   FolderName[8:11] == 'RUN' or FolderName[8:11] == 'SEG' or FolderName[8:14] == 'HYBRID':
    FACTORS = np.array(['0.03','0.04','0.06','0.07','0.10','0.13','0.18','0.24','0.32','0.42','0.56','0.75','1.00','1.33','1.78','2.37','3.16','4.22','5.62','7.50','10.00','13.34','17.78','23.71','31.62','42.17','56.23','74.99','100.00','133.35','177.83','237.14','316.23'])
    segEdge = 10;    
else:
    sys.exit()
    
#JOBS = np.array([2])
FACTORS = np.array(['10.00'])

print('Jobs')
print(len(JOBS))
from numpy import array        
print(FACTORS.size)   

twini=1220./2498560.
iny1=10.4917829/twini


iny=10



win=round(wins/twini)
print('win:')
print(win)
saltos2=array([-1.5,-1.2,-.8,-.5*wins,-.75*wins,-.25*wins])
saltos=array([-round(1.5/twini),-round(1.2/twini),-round(.8/twini),-round(win/2),-round(.75*win),-round(.25*win)])

mm=0
for i_job in np.arange(0,len(JOBS)): 
    print('                                               ')
    print('***********************************************')
    print('Job:         ' + str(i_job+1) + ' of ' + str(JOBS.size) + '  -  ' + str(JOBS[i_job]))
    
    job          = JOBS[i_job]
    GPSini = CWB_ReadLogJobs.GetGPSini(DataPath,FolderName,job)
    
    GPSiniEdge, TinjAll, WFnameAll, TinjL1All, TinjH1All = CWB_ReadLogJobs.GetInjectionTimes(DataPath,FolderName,job,GPSini)
  
    
    print('i_jobs')
    print(i_job)
     #Check segEdge
    if segEdge != GPSini-GPSiniEdge:
        print('PILAS PERRITO: problem with segEdge')
        sys.exit() 
    #print('GPSini: '+str(GPSini))
    #print('GPSiniEdge: '+str(GPSiniEdge))
    #print(' TinjAll: '+ str(TinjAll))
    Tinj                = TinjAll    - GPSiniEdge
    TinjL1              = TinjL1All  - GPSiniEdge
    TinjH1              = TinjH1All  - GPSiniEdge
    GPSiniEdge          = 0 
    
    print('# of inj:    ' + str(Tinj.size) )

#    print(str(Tinj))
    print('Times where injections of L1 ocurred' + str(TinjL1))
    print('Times where injections of H1 ocurred' + str(TinjH1))
    print('Tinj: ' +str(Tinj))
    lo=len(TinjL1)
    print(lo)
    
    
    for i_fac in np.arange(0,FACTORS.size): 
        print('Factor:      ' + str(i_fac+1) + ' of ' + str(FACTORS.size) + '  -  ' + FACTORS[i_fac] )
        fac               = FACTORS[i_fac]
        
        Rinj              = 10./float(fac)
        print('Distance:    ' + str(i_fac+1) + ' of ' + str(FACTORS.size) + '  -  ' + str(Rinj) + 'Kpc')
        
        strainH1raw, strainL1raw    = CWB_ReadStrain.ReadStrain(DataPath,FolderName,job,fac,GPSini,fs,'r',doplot)
        m=len(strainH1raw)
     
        if doprint == 1:
            print('----')
            print('Raw data duration H1: ' + str(strainH1raw.duration) + ' seconds')
            print('Raw data duration L1: ' + str(strainL1raw.duration) + ' seconds')
        
        if doplot == 1:
            
            # Choose one of the injections
            iTemp    = iny # Cuidado: este numero no puede ser mayor a Tinj.size
            
            # Imprimir nombre de la injected GW
            print('Injected GW:    ' + WFnameAll[iTemp] )
            
            # Selecccionar un intervalos de tiempo alrededor del tiempo Tinj
            t_inj    = Tinj[iTemp]
            li       = t_inj - wins/2
            lf       = t_inj + wins/2
            
            # Plot strain data
            pylab.figure(figsize=(12,4))
            pylab.plot(strainL1raw.sample_times,strainL1raw,linewidth=2.0 , label='L1')
            pylab.plot(strainH1raw.sample_times,strainH1raw,linewidth=2.0 , label='H1')
            pylab.legend()
            pylab.title('Strain around injection # ' +str(iTemp+1)+' del job '+str(job) +' ('+str(Tinj[iTemp])+')',fontsize=18)
            pylab.xlabel('Time (s)',fontsize=18,color='black')
            pylab.ylabel('Strain',fontsize=18,color='black')
            pylab.grid(True)
            
             #Plot Tinj, TinjH1, TinjL1
            for ti in Tinj:
                pylab.axvline(x=ti, color='r', linestyle='--', linewidth=1)
           # for ti in TinjL1:
            #    pylab.axvline(x=ti, color='g', linestyle=':', linewidth=1)           
           # for ti in TinjH1:
            #    pylab.axvline(x=ti, color='b', linestyle=':', linewidth=1)
            
            # Ajustar xlim al intervalo de interes
            pylab.xlim(li,lf)        

            pylab.show()
 
        high_dataL = strainL1raw.highpass_fir(16, 512)
        high_dataH = strainH1raw.highpass_fir(16, 512)
       
        whiteH = high_dataH.whiten(4, 4)
        whiteL = high_dataL.whiten(4, 4)
       
        strainL=whiteL.copy()
        strainH=whiteH.copy()

        if doplot == 1:
            
            # Choose one of the injections
            iTemp    = iny # Cuidado: este numero no puede ser mayor a Tinj.size
            
            # Imprimir nombre de la injected GW
            print('Injected GW:    ' + WFnameAll[iTemp] )
            
            # Selecccionar un intervalos de tiempo alrededor del tiempo Tinj
            t_inj    = Tinj[iTemp]
            li       = t_inj - wins/2
            lf       = t_inj + wins/2
           
            # Plot strain data
            pylab.figure(figsize=(12,4))
            pylab.plot(strainL.sample_times,strainL,linewidth=2.0 , label='L1')
            pylab.plot(strainH.sample_times,strainH,linewidth=2.0 , label='H1')
            pylab.legend()
            pylab.title('Strain around injection whitened # ' +str(iTemp+1) +' ('+str(Tinj[iTemp])+')',fontsize=18)
            pylab.xlabel('Time (s)',fontsize=18,color='black')
            pylab.ylabel('Strain',fontsize=18,color='black')
            pylab.grid(True)
            
             #Plot Tinj, TinjH1, TinjL1
            for ti in Tinj:
                pylab.axvline(x=ti, color='r', linestyle='--', linewidth=1)
           # for ti in TinjL1:
            #    pylab.axvline(x=ti, color='g', linestyle=':', linewidth=1)           
           # for ti in TinjH1:
            #    pylab.axvline(x=ti, color='b', linestyle=':', linewidth=1)
            
            # Ajustar xlim al intervalo de interes
            pylab.xlim(li,lf)        

            pylab.show()
        
        for i in range(0,366):
            m=math.floor(i/6)
            if WFnameAll[m]=='sch1':
                name.append(1)
            elif WFnameAll[m]=='sch2':
                name.append(2)
            elif WFnameAll[m]=='sch3':
                name.append(3)
                     
        
        for j in range(0,61):
            for i in range(0,6):
                li=Tinj[j]+saltos2[i]
                lf=li+wins
                ventanaL=strainL.time_slice(li, lf, mode='floor')
                ventanaH=strainH.time_slice(li,lf,mode='floor')
                # Choose one of the injections
                iTemp    =iny# Cuidado: este numero no puede ser mayor a Tinj.size
            
                # Imprimir nombre de la injected GW
                #print('Injected GW:    ' + WFnameAll[iTemp] )
            
            
            # Selecccionar un intervalos de tiempo alrededor del tiempo Tinj
              
                if j==iny:
                    # Plot strain data
                    pylab.figure(figsize=(12,4))
                    pylab.plot(ventanaL.sample_times,ventanaL,linewidth=2.0 , label='L1')
                    pylab.plot(ventanaH.sample_times,ventanaH,linewidth=2.0 , label='H1')
                    pylab.legend()
                    pylab.title('Ventana ' +str(i+1)+' sobre la inyeccion ' +str(j+1)+ ' en el intervalo: ' + str(Tinj[j]+saltos2[i])+', '+ str(Tinj[j]+saltos2[i]+wins),fontsize=18)
                    pylab.xlabel('Time (s)',fontsize=18,color='black')
                    pylab.ylabel('Strain',fontsize=18,color='black')
                    pylab.grid(True)
                # Plot Tinj, TinjH1, TinjL1
                    for ti in Tinj:
                        pylab.axvline(x=ti, color='r', linestyle='--', linewidth=1)
           # for ti in TinjL1:
            #    pylab.axvline(x=ti, color='g', linestyle=':', linewidth=1)           
           # for ti in TinjH1:
            #    pylab.axvline(x=ti, color='b', linestyle=':', linewidth=1)
            
            # Ajustar xlim al intervalo de interes
                    pylab.xlim(li,lf)        
                    
                    pylab.show()
                
                
                timesH, freqH, TQH = ventanaH.qtransform(delta_t=0.01,qrange=(8, 8),frange=(16, 512))
                timesL, freqL, TQL = ventanaL.qtransform(delta_t=0.01, qrange=(8, 8),frange=(16, 512))
                
                
                
                if j==iny:
                    
                        
                        pylab.figure(figsize=[12, 4])
                        pylab.pcolormesh(timesH, freqH, TQH**0.5)
                        pylab.xlim(li,lf)
                        pylab.title('ventanaH '+ str(i)+' de la inyeccion '+str(iny+1)+' del job '+str(job))
                        pylab.yscale('log')
                        pylab.show() 
                        
                        
                        pylab.figure(figsize=[12, 4])
                        pylab.pcolormesh(timesL, freqL, TQL**0.5)
                        pylab.xlim(li,lf)
                        pylab.title('ventanaL '+ str(i)+' de la inyeccion '+str(iny+1)+' del job '+str(job))
                        pylab.yscale('log')
                        pylab.show()     
                        
                        
            #    col=([aux[i+j*6],name[i+j*6],arr])
             #   colH=([aux[i+j*6],name[i+j*6],TQH])
              #  colL=([aux[i+j*6],name[i+j*6],TQL])
                
                if TQL.shape==(28,50):
                    if i_job==3:
                        DattestH[i+j*6]=TQH
                    else:
                        DatH[i+j*6+(i_job)*366]=TQH
                    if i_job==3:
                        DattestL[i+j*6]=TQL
                    else:
                        DatL[i+j*6+(i_job)*366]=TQL
                else:
                    mm=mm+1
                arr.append(TQH)
                arr.append(TQL)

                
np.save('DatL.npy',DatL)
np.save('DatH.npy',DatH)
np.save('DattestL.npy',DattestL)
np.save('DattestH.npy',DattestH)
print('TQH shape')
print(TQH.shape)
print('TimesH freqH')
print(timesH.shape)
print(freqH.shape)
print(timesH)

print('TQL shape')
#print(TQL)
print(TQL.shape)
print('DatL')
print(DatL.shape)
print('DatH')
print(DatH.shape)
print('DattestL')
print(DattestL.shape)
print('DattestH')
print(DattestH.shape)
print(mm)
sys.exit()



