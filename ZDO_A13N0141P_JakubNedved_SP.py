# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import skimage
import skimage.feature
dir(skimage.feature)
#import skimage.feature.hog
import skimage.data
import skimage.color
import skimage.exposure
import skimage.transform
import glob
import os
import numpy as np
#import urllib
import pickle
import sklearn
import sklearn.naive_bayes

import cv2
import skimage.io
import skimage.morphology



class Znacky:
    """
    
    #"""
    def __init__(self):
        # Toto mi umožňuje zapínat a vypínat různé části příznakového vektoru
        self.grayLevelFeatures = True
        self.colorFeatures = False  # rozpoznávání podle barvy
        self.hogFeatures = False
        self.labels = None

        # Načítání natrénovaných parametrů klasifikátoru ze souboru atd.
        path_to_script = os.path.dirname(os.path.abspath(__file__))
        classifier_path = os.path.join(path_to_script,
                                       "data.pkl")
        try:
            saved = pickle.load(open(classifier_path,  "rb"))
            self.clf = saved[0]
            self.labels = saved[1]
            # Ukazka logovani
        except:
            print 'problem se vstupnim souborem'
        pass
    


    def one_file_features(self, im, demo=True):
    
        """
        Zde je kontruován vektor příznaků pro klasfikátor
        """
        fd = np.array([])
        
       
        #%% Změna velikosti
        import skimage.transform
        import skimage
        
        
        image = skimage.transform.resize(im, [50, 50])

        #%% Vyriznuti objektu kolem stredu obrazu
        image = image[int(image.shape[0]/2)-20:int(image.shape[0]/2)+20, int(image.shape[1]/2)-20:int(image.shape[1]/2)+20]
        fd = np.append(fd, image.reshape(-1))
        from skimage import filter

        camera = image[:]
        val = filter.threshold_otsu(camera)
        TF = camera < val
        
        fd = np.append(fd, TF.reshape(-1))
        #fd.append(hsvft[:])
        #if self.colorFeatures:
        #    fd = np.append(fd, self.colorFeatures)
        #    pass

        return fd

    # nacitani z adresare
    def readImageDir(self, path):
        dirs = glob.glob(os.path.join(os.path.normpath(path), '*'))
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

    def train(self, datadir='../zdo2014-training3/'):
        tfiles, tlabels = self.readImageDir(datadir)

        # trénování by trvalo dlouho, tak si beru jen každý stý obrázek
        tfiles = tfiles[::20]
        tlabels = tlabels[::20]

        featuresAll = []
        i = 0

        for fl in tfiles:
            i = i + 1
            im = skimage.io.imread(fl)
            fv = self.one_file_features(im)
            print 'Soubor c. ',i, ' Pocet priznaku: ', size(fv)
            featuresAll.append(fv)

        featuresAll = np.array(featuresAll)
        print 'Vse precteno, jde se trenovat'
        
        # Trénování klasifikátoru

        labels, inds = np.unique(tlabels, return_inverse=True)
        
        #from sklearn import svm
        #clf = svm.SVC()
        clf = sklearn.naive_bayes.GaussianNB()

        clf.fit(featuresAll, inds)
        self.clf = clf
        self.labels = labels

        # ulozime do souboru pomocí modulu pickle
        # https://wiki.python.org/moin/UsingPickle

        # je potřeba zachovat i původní labely
        saved = [clf, labels]
        pickle.dump(saved, open("data.pkl", "wb"))

    def rozpoznejZnacku(self, image, demo=False):

        # Nějaký moc chytrý kód

        class_index = self.clf.predict(self.one_file_features(image, demo))
        # tady převedeme číselnou hodnotu do textového popisku
        retval = self.labels[class_index]

        return retval[0]
        
    def kontrola(self, datadir=None):
        import time
        #obrazky = ['http://147.228.240.61/zdo/P2_id14368_ff74-FL_1_131030_00002530.jpg',
        #         'http://147.228.240.61/zdo/Z3_id18972_ff2347-FL_1_131030_00020439.jpg',
        #         'http://147.228.240.61/zdo/P1_id13258_ff7546-FL_1_131030_00066180-1.jpg'
        #         ]
        #reseni = ['P2', 'Z3', 'P1']
        
        #datadir='../zdo2014-training1/'
             
        obrazky, reseni = self.readImageDir(datadir)
        return obrazky
        vysledky = []
        
        for i in range(0, len(obrazky)):
            cas1 = time.clock()
            im = skimage.io.imread(obrazky[i])
            result = self.rozpoznejZnacku(im)           
    
            cas2 = time.clock()   
    
            if((cas2 - cas1) >= 1.0):
                print "cas vyprsel"
                result = 0
    
            vysledky.append(result)
                
        hodnoceni = np.array(reseni) == np.array(vysledky)
        skore = np.sum(hodnoceni.astype(np.int)) / np.float(len(reseni))
        
        print skore

# <codecell>

# následující zápis zařídí spuštění při volání z příkazové řádky.
# Pokud bude modul jen includován, tato část se nespustí. To je požadované
# chování
if __name__ == "__main__":
    zn = Znacky()
    zn.train()
    
    #zn.kontrola('../zdo2014-training3/')

# <codecell>

#print clf
        
