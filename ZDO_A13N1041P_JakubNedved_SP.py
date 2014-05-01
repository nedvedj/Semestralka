# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import matplotlib.pyplot as plt
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


class Znacky:
    """
    M. Jiřík
    I. Pirner
    P. Zimmermann
    Takto bude vytvořeno vaše řešení. Musí obsahovat funkci
    'rozpoznejZnacku()', která má jeden vstupní parametr. Tím je obraz. Doba
    trváná funkce je omezena na 1 sekundu. Tato funkce rovněž musí obsahovat 
    ukázkový režim. V něm je pomocí obrázků vysvětleno, jak celá věc pracuje.
    #"""
    def __init__(self):
        # Toto mi umožňuje zapínat a vypínat různé části příznakového vektoru
        self.grayLevelFeatures = True
        self.colorFeatures = False  # rozpoznávání podle barvy
        self.hogFeatures = False
        self.labels = None

        # Načítání natrénovaných parametrů klasifikátoru ze souboru atd.
        # path_to_script = os.path.dirname(os.path.abspath(__file__))
        classifier_path = os.path.join(path_to_script,
                                       "data.pkl")
        try:
            saved = pickle.load(open(classifier_path,  "rb"))
            self.clf = saved[0]
            self.labels = saved[1]
        except:
            print "Problems with file " + "data.pkl"
        pass

    def one_file_features(self, im, demo=True):
    
        """
        Zde je kontruován vektor příznaků pro klasfikátor
        """
        # color processing
        fd = np.array([])
        im = im>150
        img = skimage.color.rgb2gray(im)
        # graylevel
        if self.hogFeatures:
            pass

        if self.grayLevelFeatures:
            imr = skimage.transform.resize(img, [10, 10])
            glfd = imr.reshape(-1)
            fd = np.append(fd, glfd)
            
            if demo:
                plt.imshow(imr)
                plt.show()

        #fd.append(hsvft[:])
        if self.colorFeatures:
            fd = np.append(fd, self.colorFeatures)
            pass

        #print hog_image
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
            print i
            im = skimage.io.imread(fl)
            fv = self.one_file_features(im)
            featuresAll.append(fv)

        featuresAll = np.array(featuresAll)

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
