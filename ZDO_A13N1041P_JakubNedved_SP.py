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
import urllib



class Znacky:
    """
    
    #"""
    def __init__(self, params_online=True):
        self.colorFeatures = False
        self.hogFeatures = False
        self.grayLevelFeatures = True
        
        # Načítání natrénovaných parametrů klasifikátoru ze souboru atd.
        if params_online:
            url = 'https://raw.githubusercontent.com/nedvedj/Semestralka/master/data.pkl'
            urllib.urlretrieve(url, "data.pkl")
            
        try:
            self.clf = pickle.load( open( "data.pkl", "rb" ) )
        except:
            print "Problems with file " + "data.pkl"
        pass
    


    def one_file_features(self, image, demo=True):
    
        """
        Zde je kontruován vektor příznaků pro klasfikátor
        """
  
        pole = np.array([])

        #if self.colorFeatures:
        #    pole = np.append(pole, self.colorFeatures)
            
        #%% Prevod RGB2GRAY pro detekci hran a dalsi zpracovani
        cernobily = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        #%% Změna velikosti
        image = skimage.transform.resize(image, [50, 50])
        
     
        #%% Vyriznuti objektu kolem stredu obrazu
        image = image[int(image.shape[0]/2)-20:int(image.shape[0]/2)+20, int(image.shape[1]/2)-20:int(image.shape[1]/2)+20] 
        pole = np.append(pole, image.reshape(-1))
        
        #%% Detekce hran
        
        cernobily = cv2.Canny(cernobily,60,60)
        
        kernel_big = skimage.morphology.diamond(1) 
        cernobily = skimage.morphology.binary_dilation(cernobily, kernel_big) # Na detekovane hrany pouzijeme dilataci
        cernobily = cv2.GaussianBlur(cernobily,(5,5), 5) # Gausovska filtrace pro odstraneni nezadoucich objektu
        
        
        #%% Vyriznuti objektu kolem stredu obrazu
        cernobily = cernobily[int(cernobily.shape[0]/2)-20:int(cernobily.shape[0]/2)+20, int(cernobily.shape[1]/2)-20:int(cernobily.shape[1]/2)+20]
       
        pole = np.append(pole, cernobily.reshape(-1))
        
    
        return pole

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

# <codecell>

# následující zápis zařídí spuštění při volání z příkazové řádky.
# Pokud bude modul jen includován, tato část se nespustí. To je požadované
# chování
if __name__ == "__main__":
    zn = Znacky()
    zn.train()


# <codecell>

#print clf
