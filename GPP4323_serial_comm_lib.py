# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:14:37 2023

A very simple library to control GW Instek GPP-4323 Voltage/Current generator.
The device has to be set in remote operation mode:
    System - Interface - USB - 9600

Working example:
    
    >>> from GH.labtools.labtools.GWInstek.GPP4323 import GWInstekController
    
    >>> GW = GWInstekController()
    >>> GW.idn()
    
    >>> GW.set_voltage(channel=CHANNEL, voltage=0.1) 
    >>> GW.enable_output(channel=CHANNEL)
    >>> measured_voltage = GW.get_current(channel=CHANNEL)
    
    >>> GW.disable_output(channel=1)
    >>> GW.close()

Remember to CLOSE the port in the end!
If a crash occurs, unlock and reconnect the cable.

@author: Simon
"""

import serial
import datetime
import serial.tools.list_ports
import time
import numpy as np
#import pyTENMA


# set port and channel
# FOLDER = ""
# timestamp = str(datetime.datetime.now()).replace(":", ".")[:-7]
# PORT0 = 'COM5'  # device manager
# CHANNEL = 1

   


class GWInstekController(object):
    
    def __init__(self, PORT):
        self._serial = serial.Serial(port = PORT, baudrate=9600, timeout = 1)
        self.port = PORT
        
    def idn(self):
        self._serial.write(b'*IDN?\n')
        response = self._serial.readline()
        print(response.decode('UTF-8'))
        
    def beeper_off(self):
        self._serial.write(b'BEEP0\n')

    def beeper_on(self):
        self._serial.write(b'BEEP1\n')

    def disable_output_all(self):
        self._serial.write(b'OUT0\n')
        
    def enable_output(self, channel):
        self._serial.write(b':OUTP%i 1\n' % (channel))
        
    def disable_output(self, channel):
        self._serial.write(b':OUTP%i 0\n' % (channel))
    
    def set_current(self, channel, current):
        self._serial.write(b'ISET%i:%f\n' % (channel, current))
        #self._serial.write('OUT1\n')
    
    def get_current(self, channel):
        self._serial.write(b'IOUT%i?\n' % channel)
        return float(self._serial.readline()[0:6])


    def set_voltage(self, channel, voltage):
        self._serial.write(b'VSET%i:%f\n' % (channel, voltage))

    
    def get_voltage(self, channel):
        self._serial.write(b'VOUT%i?\n' % channel)
        return float(self._serial.readline()[0:6])

    def close(self):
        self.disable_output_all()
        self._serial.close()
