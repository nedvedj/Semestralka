# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:14:26 2014

@author: nedvedj
"""

import signal

import skimage
import skimage.io
import numpy as np

import os
path_to_script = os.path.join(os.path.dirname(__file__))
# na windows nefunguje knihovna contextlib
# v kodu je proto náhrada od M. Červeného pomocí time
import time

        
def kontrola(ukazatel):
    studentske_reseni = ukazatel() # tim je zavolán váš konstruktor __init__
    
    obrazky = ['http://147.228.240.61/zdo/P2_id14368_ff74-FL_1_131030_00002530.jpg',
             'http://147.228.240.61/zdo/Z3_id18972_ff2347-FL_1_131030_00020439.jpg',
             'http://147.228.240.61/zdo/P1_id13258_ff7546-FL_1_131030_00066180-1.jpg'
             ]
    reseni = ['P2', 'Z3', 'P1']
    
    vysledky = []
    
    for i in range(0, len(obrazky)):
        cas1 = time.clock()
        im = skimage.io.imread(obrazky[i])
        result = studentske_reseni.rozpoznejZnacku(im)           

        cas2 = time.clock()   

        if((cas2 - cas1) >= 1.0):
            print "cas vyprsel"
            result = 0

        vysledky.append(result)
            
    hodnoceni = np.array(reseni) == np.array(vysledky)
    skore = np.sum(hodnoceni.astype(np.int)) / np.float(len(reseni))
    
    print skore
    
ukazatel = Znacky
kontrola(ukazatel)