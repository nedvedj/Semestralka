
import skimage
import skimage.io
import numpy as np


import time

import os
import glob

def readImageDir(path=None):
        dirs = glob.glob(os.path.join(os.path.normpath(path), '*'))
        #print dirs
        labels = []
        #nlabels = []
        files = []

        #i = 0
        for onedir in dirs:

            #print onedir
            base, lab = os.path.split(onedir)
            if os.path.isdir(onedir):
                filesInDir = glob.glob(os.path.join(onedir, '*'))
                for onefile in filesInDir:
                    labels.append(lab)
                    files.append(onefile)
                    #nlabels.append(i)

        return files, labels
    

def kontrola(ukazatel,data):
    studentske_reseni = ukazatel() 
    """
    cesta = "../zdo2014-training/"
    obrazky = ['P1/P1_id13063_ff3512-FL_1_131030_00035316.jpg',
             'P1/P1_id13064_ff3513-FL_1_131030_00035322.jpg',
             'P1/P1_id13339_ff8126-FL_1_131030_00070898.jpg',
             'P2/P2_id14356_ff40-131030_00002105.jpg',
             'P2/P2_id14368_ff67-FL_1_131030_00002463-1.jpg',
             'P2/P2_id14375_ff88-FL_1_131030_00002661-2.jpg',
             'Z3/Z3_id18848_ff2515-131030_00029137.jpg',
             'Z3/Z3_id18849_ff2523-131030_00029216.jpg',
             'Z3/Z3_id18970_ff2345-FL_1_131030_00020419-1.jpg'
             ]
    reseni = ['P1', 'P1', 'P1', 'P2', 'P2', 'P2', 'Z3', 'Z3', 'Z3']
    """
    
    #datadir='../zdo2014-training1/'
         
    obrazky, reseni = readImageDir('../zdo2014-training3/')
    
    #obrazky = obrazky[::data]
    #reseni = reseni[::data]
    #velikost = len(obrazky)
    #obrazky = obrazky[int(velikost/5*4):]
    #reseni = reseni[int(velikost/5*4):]

    vysledky = []
    
    for i in range(0, len(obrazky)):
        cas1 = time.clock()
        #im = skimage.io.imread(cesta+obrazky[i])
        #print obrazky[i]
        try:
            im = skimage.io.imread(obrazky[i])
        except IOError:
            continue
        if i==1000: print 'Soubor c. ',i
            
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

for data in [12,16,20]:
    kontrola(ukazatel,data)


