#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

class PT_1:
    def __init__(self, T, dt, K, last):
        self.T = T
        self.dt = dt
        self.K = K
        self.last_out = last
        
    def get_output(self, regelwert):
        output = (self.T/self.dt+ 1)**(-1) * (self.K * regelwert - self.last_out) +self.last_out
        self.last_out = output
        return output
    
    
class PID:
    def __init__(self, K_p, T_n, T_v, dt):
        self.K_p = K_p
        self.T_n = T_n
        self.T_v = T_v
        self.dt = dt
        self.last_i_val = 0
        self.last_val = 0

    def get_controll_val(self, soll, ist):
        #Regler
        diff = (soll-ist)
        p_val = diff * self.K_p
        i_val = self.last_i_val + diff * self.dt / self.T_n
        d_val = self.T_v * (ist - self.last_val) / self.dt
        #print(p_val)
        
        #letzte Werte speichern
        self.last_i_val = i_val
        self.last_val = ist
        if self.T_n == 0 and self.T_v == 0:
            return p_val #+ i_val + d_val #
        elif self.T_n == 0:
            return p_val + d_val #
        elif  self.T_v == 0:
            return p_val + i_val 
        else:
            return p_val + i_val - d_val 



def sollwert(t):
    return  np.sin(t)+1

def regler(soll, s, v, a):
    Kp=0.5
    return (soll-s)*Kp



def simulation():
    return 0

def main():
    #Einstellungen
    simulationsdauer = 30
    simulationsschritte = 10000
    eingriffsverhaeltnis = 10
    
    #Startwerte
    stellwert = 0
    time, dt = np.linspace(0, simulationsdauer, simulationsschritte, retstep=True)
    
    #Objekte
    myPID = PID(0.5, 0, 0, 10*dt)
    myPT_1_1 = PT_1(0.1, dt, 5, stellwert)    
    
    winkel = []
    winkel_ges = []
    
    for t in time[::eingriffsverhaeltnis]:
        regelwert = myPID.get_controll_val(sollwert(t), stellwert)
        for k in np.arange(1, eingriffsverhaeltnis+1):                          #Regler im richtigen Verhältnis aufrufen
            stellwert =myPT_1_1.get_output(regelwert)
            winkel.append(stellwert)
            winkel_ges.append(sollwert(t))

    plt.plot(time, winkel)
    plt.plot(time, winkel_ges)
    plt.gca().set_ylim(-1,3)
    plt.grid()
    print(winkel[-1])
    plt.show()
    print(t.shape)


if __name__ == "__main__":
    main()