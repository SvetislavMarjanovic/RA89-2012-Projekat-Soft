# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:16:44 2016

@author: Svetislaav Marjanovic
"""

#import potrebnih biblioteka za K-means algoritam

#---------------IMPORT
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

#Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans

#import potrebnih biblioteka
import cv2
import collections

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12 # za prikaz veCih slika i plotova, zakomentarisati ako nije potrebno

from Tkinter import*
from tkFileDialog import askopenfilename
 

import ttk
import PIL
from PIL import Image, ImageTk, ImageDraw


#from __future__ import print_function
#print("hi there", file=f)


#---------------FUNKCIJE
#Funkcionalnost implementirana u V1
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((2,2)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((2,2)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)
def erodeStronger(image):
    kernel = np.ones((6,6)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)
def adaptive(imageBinarna):
    image=cv2.adaptiveThreshold(imageBinarna, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 30, 2)
    return image
    
    # TODO 6
def select_roi(image_orig, image_bin):
 
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_dict = {}
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        if(w>5 and h>12):
            region = image_bin[y:y+h+1,x:x+w+1];
            regions_dict[x] = [resize_region(region), (x,y,w,h)]
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),1)

    sorted_regions_dict = collections.OrderedDict(sorted(regions_dict.items()))
    sorted_regions = np.array(sorted_regions_dict.values())
    
    sorted_rectangles = sorted_regions[:,1]
    region_distances = [-sorted_rectangles[0][0]-sorted_rectangles[0][2]]
    for x,y,w,h in sorted_regions[1:-1, 1]:
        region_distances[-1] += x
        region_distances.append(-x-w)
    region_distances[-1] += sorted_rectangles[-1][0]
    
    return image_orig, sorted_regions[:, 0], region_distances
    
#njihova metoda za prikaze rezultata rada neuronske mreze
def display_result(outputs, alphabet):
   
   # w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    # result = alphabet[winner(outputs[0])]
     #for idx, output in enumerate(outputs[1:,:]):d
       
        # if (k_means.labels_[idx] == w_space_group):
        #    result += ' '
        #result += alphabet[winner(output)]
    # return result
    
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result
#Funkcionalnost implementirana u V2
def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized
def scale_to_range(image):
    return image / 255
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]
def create_ann():
    '''
    Implementirati veStaCku neuronsku mreZu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    # Postaviti slojeve neurona mreze 'ann'
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(40, activation='sigmoid'))
    return ann
    
def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, nb_epoch=2000, batch_size=1, verbose = 0, shuffle=False, show_accuracy = False) 
      
    return ann
    

def openFile(ann, alphabet):
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    location = askopenfilename() # show an "Open" dialog box and return the path to the selected file
  #  radiSve(location)
    dugmeSacuvaj = Button(root, text="Upisi podatke", width=20, height=5, command=lambda:prepoznavanje(ann, alphabet))
    dugmeSacuvaj.place(x=530,y=500)
    
    return location
   
   
def set_Pic_path(text):
    img = Image.open(text)
    img = img.resize((675, 430), PIL.Image.ANTIALIAS)
    
    img.save('resized.jpg')
  
    photo = ImageTk.PhotoImage(Image.open('resized.jpg'))
    labelSlika = Label(root, image=photo)
    labelSlika.image = photo
    labelSlika.place(x=20, y=20)
       
   
    return


def display_result_di(outputs, alphabet, k_means):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        if (k_means.labels_[idx] == w_space_group):
            result += ' '
            
        if (alphabet[winner(output)] == "0"):
            result += "O"
        elif (alphabet[winner(output)] == "8"):
            result += "V"
        else:
            result += alphabet[winner(output)]
            
    return result

def kmeans_di(razdaljina):
    #k means
    #Podešavanje centara grupa K-means algoritmom
    distances = np.array(razdaljina).reshape(len(razdaljina), 1)
    #Neophodno je da u K-means algoritam bude prosleđena matrica u kojoj vrste određuju elemente
    
    k_means = KMeans(n_clusters=2, max_iter=2200, tol=0.00001, n_init=10)
    k_means.fit(distances)

    return k_means
    
    
    
    
def obucavanjeNM():
    print "USAO U OBUCAVANJE"
    
    imageTrain = load_image('imagesprimeri/OSmmbw.jpg')
    imageTrainGray = image_gray(imageTrain)
    imageTraingBin = invert(image_bin(imageTrainGray))
    imageTrainCopy = imageTrain.copy()
    imageTrainCopy, regioniTrain, razdaljinaTrain = select_roi(imageTrainCopy, imageTraingBin)
    
    #alphabet = ['A','V','G','D','E','Ž','I','J','K','L','M','N','O','P','R','S','T','Ć','U','H','Š','C','Č','LJ','B','F','NJ','DŽ','0','1','2','3','4','5','7','8','9','6']
    alphabet = ['A','B','V','G','D','DJ','E','Ž', 'Z','I','J','K','L','LJ', 'M','N','NJ','O','P','R','S','T','Ć','U','F','H','C','Č','DŽ','Š','0','1','2','3','4','5','6','7','8','9']
    #priprema izdvojenih regiona sa slike za ucenje a potom i sa ciljane slike
    inputs = prepare_for_ann(regioniTrain)
    
    outputs = convert_output(alphabet)
    #kreiranje neuronske mreze
    ann = create_ann()
    ann = train_ann(ann, inputs, outputs)
    
    print "ZAVRSIO OBUCAVANJE"

    return ann, alphabet
    
def prepoznavanje(ann, alphabet):
    
    print "USAO U PREPOZNAVANJE"
    
    image = load_image('resized.jpg')
    imageLK = cv2.resize(image,(675,430), interpolation = cv2.INTER_NEAREST)
    imageLKGray = image_gray(imageLK)
    imageLKBin = image_bin(imageLKGray)
    imageSecena = imageLKBin[100:360,205:530] #isecena slika BIN
    imageSecenaPrava = imageLK[100:360,205:530]

    imageSecena2 = imageLKBin[360:415, 0:350]

    
    imageLKCopy = imageSecenaPrava.copy()
    imageLKCopy, regioniLK, razdaljinaPravee = select_roi(imageSecenaPrava, invert(imageSecena))
    
    imageSecenaCopy = imageSecena.copy()
    height,width=imageSecenaCopy.shape[0:2]
    
    imageIsecenaIme = imageSecenaCopy.copy()[0:45, 0:width]
    imageSI = imageSecenaPrava.copy()[0:45, 0:width]
    
    imageIsecenaImeCopy = imageIsecenaIme.copy()
 #   imageIsecenaImeCopy, regioniIme, razdaljinaIme = select_roi(imageSI, invert(imageIsecenaIme))
    imageIsecenaImeCopy, regioniIme, razdaljinaIme = select_roi(imageIsecenaImeCopy, invert(imageIsecenaIme))
    
    
    imageIsecenaPrezime = imageSecenaCopy.copy()[45:90, 0:width]
    imageIsecenaPrezimeCopy = imageIsecenaPrezime.copy()
    imageIsecenaPrezimeCopy, regioniPrezime, razdaljinaPrezime = select_roi(imageIsecenaPrezimeCopy, invert(imageIsecenaPrezime))
    
    
    imageIsecenaDatumRodjenja = imageSecenaCopy.copy()[90:130, 0:width]
    imageIsecenaDatumRodjenjaCopy = imageIsecenaDatumRodjenja.copy()
    imageIsecenaDatumRodjenjaCopy, regioniDatumRodjenja, razdaljinaDatumRodjenja = select_roi(imageIsecenaDatumRodjenjaCopy, invert(imageIsecenaDatumRodjenja))
    
    
    imageIsecenaPolRegBr = imageSecenaCopy.copy()[130:175, 0:width]
    imageIsecenaPolRegBrCopy = imageIsecenaPolRegBr.copy()
    imageIsecenaPolRegBrCopy, regioniPolRegBr, razdaljinaPolRegBr = select_roi(imageIsecenaPolRegBrCopy, invert(imageIsecenaPolRegBr))
    
    
    imageIsecenaDatumIzdavanja = imageSecenaCopy.copy()[175:220, 0:width]
    imageIsecenaDatumIzdavanjaCopy = imageIsecenaDatumIzdavanja.copy()
    imageIsecenaDatumIzdavanjaCopy, regioniDatumIzdavanja, razdaljinaDatumIzdavanja = select_roi(imageIsecenaDatumIzdavanjaCopy, invert(imageIsecenaDatumIzdavanja))
    
    
    
    imageIsecenaVaziDo = imageSecenaCopy.copy()[220:260, 0:width]
    imageIsecenaVaziDoCopy = imageIsecenaVaziDo.copy()
    imageIsecenaVaziDoCopy, regioniVaziDo, razdaljinaVaziDo = select_roi(imageIsecenaVaziDoCopy, invert(imageIsecenaVaziDo))
    
    
    imageIsecenaDokumentIzdaje = imageSecena2.copy()
    imageIsecenaDokumentIzdajeCopy = imageIsecenaDokumentIzdaje.copy()
    imageIsecenaDokumentIzdajeCopy, regioniDokumentIzdaje, razdaljinaDokumentIzdaje = select_roi(imageIsecenaDokumentIzdajeCopy, invert(imageIsecenaDokumentIzdaje))
    
    
    
    plt.figure()
    display_image(imageIsecenaImeCopy)
    
    plt.figure()
    display_image(imageIsecenaPrezimeCopy)
    
    plt.figure()
    display_image(imageIsecenaDatumRodjenjaCopy)
    
    plt.figure()
    display_image(imageIsecenaPolRegBrCopy)
    
    plt.figure()
    display_image(imageIsecenaDatumIzdavanjaCopy)
    
    plt.figure()
    display_image(imageIsecenaVaziDoCopy)
    
    plt.figure()
    display_image(imageIsecenaDokumentIzdajeCopy)
    
    plt.figure()
    display_image(image)
    
    inputs3 = prepare_for_ann(regioniIme)
    inputs4 = prepare_for_ann(regioniPrezime)
    inputs5 = prepare_for_ann(regioniDatumRodjenja)
    inputs6 = prepare_for_ann(regioniPolRegBr)
    inputs7 = prepare_for_ann(regioniDatumIzdavanja)
    inputs8 = prepare_for_ann(regioniVaziDo)
    inputs9 = prepare_for_ann(regioniDokumentIzdaje)
    
    results = ann.predict(np.array(inputs3, np.float32))
    results4 = ann.predict(np.array(inputs4, np.float32))
    results5 = ann.predict(np.array(inputs5, np.float32))
    results6 = ann.predict(np.array(inputs6, np.float32))
    results7 = ann.predict(np.array(inputs7, np.float32))
    results8 = ann.predict(np.array(inputs8, np.float32))
    results9 = ann.predict(np.array(inputs9, np.float32))
    
    #results3 = ann.predict(np.array(inputs3, np.float32))
    
    prezime= display_result(results, alphabet)
    ime= display_result(results4, alphabet)
    datumRodjenja= display_result(results5, alphabet)
    polRBr= display_result(results6, alphabet)
    datumIzdavanja= display_result(results7, alphabet)
    vaziDo= display_result(results8, alphabet)
    dokumentIzdaje= display_result_di(results9, alphabet, kmeans_di(razdaljinaDokumentIzdaje))

  #  print "Tekst ... : "
  #  print prezime 
  #  print " - "
  #  print ime
  #  print " - "
  #  print datumRodjenja
  #  print " - "
  #  print polRBr
  #  print " - "
  #  print datumIzdavanja
  #  print " - "
  #  print vaziDo
    
    
    brojacDatumRodjenja = 0
    brojacPolRbr = 0
    brojacDatum = 0
    brojacDatum2 = 0
    
    string = ''
    tab = ''
    
    #PREZIME
    for x in prezime:
        if(x == "0"):
            x = "O"
            string += x
        elif(x == "8"):
            x = "V"
            string += x
        else:
            string += x
            
    string += ","   
    if(len(string)<7):
        string += "\t\t\t"
    else:
        string += "\t\t"
    
    
    
    #IME
    for x in ime:
        if(x == "0"):
            x = "O"
            string += x
            tab += x
        elif(x == "8"):
            x = "V"
            string += x
            tab += x
        else:
            string += x
            tab += x
        
    string += ","
    if(len(tab)<7):
        string += "\t\t\t"
    else:
        string += "\t\t"
    
    #DATUM RODJENJA
    for x in datumRodjenja:
        brojacDatumRodjenja =brojacDatumRodjenja + 1
        if(brojacDatumRodjenja == 2 or brojacDatumRodjenja == 4):
            if(x == "O"):
                x = "0"
                string += x + "."
            else:
                string += x + "."
        else:
            if(x == "O"):
                x = "0"
                string += x
            else:
                string += x
    string += ".,"
    string += "\t\t\t"
    
    
    #POL I BROJ LICNE KARTE
    for x in polRBr:
        brojacPolRbr = brojacPolRbr + 1
        if(brojacPolRbr == 1):
            string += x + "    "
        else:
            if(x == "O"):
                x = "0"
                string += x
            elif(x == "Z"):
                x = "3"
                string += x
            elif(x == "V"):
                x = "8"
                string += x
            elif(x == "E"):
                x = "5"
                string += x
            else:
                string +=x
    string += ","
    string += "\t\t"
    
    
    #DATUM IZDAVANJA
    for x in datumIzdavanja:
        brojacDatum = brojacDatum + 1
        if(brojacDatum == 2 or brojacDatum == 4):
            if(x == "O"):
                x = "0"
                string += x + "."
            elif(x == "V"):
                x = "8"
                string += x + "."
            elif(x == "E"):
                x = "5"
                string += x + "."
            else:
                string += x + "."
        else:
            if(x == "O"):
                x = "0"
                string += x
            elif(x == "V"):
                x = "8"
                string += x
            elif(x == "E"):
                x = "5"
                string += x
            else:
                string += x
    string += ".,"    
    string += "\t\t"
    
    
    #DATUM VAZENJA
    for x in vaziDo:
        brojacDatum2 = brojacDatum2 + 1
        if(brojacDatum2 == 2 or brojacDatum2 == 4):
            if(x == "O"):
                x = "0"
                string += x + "."
            elif(x == "V"):
                x = "8"
                string += x + "."
            elif(x == "E"):
                x = "5"
                string += x + "."
            else:
                string += x + "."
        else:
            if(x == "O"):
                x = "0"
                string += x
            elif(x == "V"):
                x = "8"
                string += x
            elif(x == "E"):
                x = "5"
                string += x
            else:
                string += x
    
    string += "."
    string += "\t\t"
    
    #IZDAJE
    string += dokumentIzdaje
 
    print "---"
    print string
    print "---"
    
    #print string
   
    
    f = open('Spisak prepoznatih osoba','a')
    f.write(string + '\n') # python will convert \n to os.linesep
    f.close() # you can omit in most cases as the destructor will call it
    
    print "ZAVRSIO PREPOZNAVANJE"

    return


#------------------------GUI




location = ''

ann, alphabet = obucavanjeNM()
root = Tk()
root.resizable(0,0)
root.title("Aplikacija za preuzimanje podataka sa licne karte RS")
root.geometry('720x600+200+200')
 
fontBold = ('Arial', 12, 'bold')
fontSerial = ('Arial', 12)
fontTextField = ('Arial', 16)

dugmeUcitaj = Button(root, text="Izberite sliku", width=20, height=5, command=lambda:set_Pic_path(openFile(ann,alphabet)))
dugmeUcitaj.place(x=30,y=500)   


root.mainloop()




