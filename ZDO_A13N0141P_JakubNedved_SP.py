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

import skimage.io
import skimage.morphology

import time



class Znacky:
    """

    #"""
    def __init__(self):
        # Toto mi umožňuje zapínat a vypínat různé části příznakového vektoru

        self.labels = None

        # Načítání natrénovaných parametrů klasifikátoru ze souboru atd.
        path_to_script = os.path.dirname(os.path.abspath(__file__))
        classifier_path = os.path.join(path_to_script,
                                       "data.pkl")
        try:
            saved = pickle.load(open(classifier_path,  "rb"))
            self.clf = saved[0]
            self.labels = saved[1]
        except:
    ValueError: need more than 1 value to unpack        print 'problem se vstupnim souborem'
        pass



    def one_file_features(self, im, demo=False):

        """
        Zde je kontruován vektor příznaků pro klasfikátor
        """
        fd = np.array([])

        import skimage.transform
        import skimage
        #import cv2

        # Zmena velikosti obrazku
        image = skimage.transform.resize(im, [40, 40])
        #image = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        #%% Vyriznuti objektu kolem stredu obrazu
        image = image[int(image.shape[0]/2)-10:int(image.shape[0]/2)+15, int(image.shape[1]/2)-10:int(image.shape[1]/2)+15]
        #image=image[::5]
        fd = np.append(fd, image.reshape(-1))

        # Vyuziti Otsuova filtru
        from skimage import filter

        threshold = filter.threshold_otsu(image)
        #image=image[::5]
        image =image < threshold

        fd = np.append(fd, image.reshape(-1))

        #%% Změna velikosti


        #fd.append(hsvft[:])
       # if self.colorFeatures:
       #     fd = np.append(fd, self.colorFeatures)
           # pass

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

        # po kolika souborech budu trenovat
        tfiles = tfiles[::5]
        tlabels = tlabels[::5]


        featuresAll = []
        i = 0

        for fl in tfiles:
            i = i + 1
            try: # pokud slozka obsahuje soubory, ktere nejsou jpg (.db)
                im = skimage.io.imread(fl)
            except IOError:
                continue
           # im = skimage.io.imread(fl)
            fv = self.one_file_features(im)
            print 'Soubor c. ',i, ' Pocet priznaku: ', len(fv)
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

        class_index = self.clf.predict(self.one_file_features(image, demo))
        retval = self.labels[class_index]

        return retval[0]


    def kontrola(self, datadir=None):
        """
        Jednoduché vyhodnocení výsledků
        """

        obrazky, reseni = self.readImageDir(datadir)

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
# následující zápis zařídí spuštění při volání z příkazové řádky.
# Pokud bude modul jen includován, tato část se nespustí. To je požadované
# chování
if __name__ == "__main__":
    import sys
    zn = Znacky()
    if len(sys.argv) < 3:
        zn.train()
    else:

        #zn.train(datadir='/home/mjirik/data/zdo2014/zdo2014-training3/')
        #zn.kontrola('/home/mjirik/data/zdo2014/zdo2014-training1/')
        print "trenovani"
        zn.train(datadir=sys.argv[1])
        print "kontrola"        
        zn.kontrola(sys.argv[2])
