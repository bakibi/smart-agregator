import tensorflow as tf
import numpy as np
import pandas as pd
from SvnGaussiKernel import SvnGaussiKernel



class batch_algo(object):
    """docstring for ."""
    def __init__(self,MAX_SEQUENCE,ALL_X,ALL_Y,BATCH_SIZE,NUMBER_OF_CATEGORIES,NUMBER_OF_CLASSIFIER):
        self.max_sequence = MAX_SEQUENCE
        self.all_x = ALL_X
        self.all_y = ALL_Y
        self.batch_size = BATCH_SIZE
        self.number_of_categories = NUMBER_OF_CATEGORIES
        self.number_classifier = NUMBER_OF_CLASSIFIER
        self.classifiers = []
        for i in range(NUMBER_OF_CLASSIFIER):
            rand_index = np.random.choice(len(ALL_X), size=self.batch_size)
            rand_x = ALL_X[rand_index]
            rand_y = ALL_Y[:, rand_index]
            self.addClassifier(featuresT=rand_x,labelsT=rand_y)
        print(len(self.classifiers))

    def addClassifier(self,featuresT,labelsT):
        model = SvnGaussiKernel(MAX_SEQUENCE=self.max_sequence,ALL_X=featuresT,ALL_Y=labelsT,BATCH_SIZE=self.batch_size,NUMBER_OF_CATEGORIES=self.number_of_categories)
        self.classifiers.append(model)

    def trainClassifier(self,sess, ind,epoch):
        print('class : ',ind    )
        self.classifiers[ind].train(sess=sess,epoch=epoch)

    def trainAllClassifers(self,sess, EPOCH):
        for i in range(self.number_classifier):
            self.trainClassifier(ind=i,sess=sess,epoch=EPOCH)

    def getPredictionFromClassifier(self,sess,ind,test_in):
        out = self.classifiers[ind].predict(sess=sess,grid=test_in)
        return out

    def getPredict(self,sess,data):
        tt = len(data)
        aa = np.zeros((tt,self.number_of_categories), dtype=int)
        for i in range(self.number_classifier):
            out = self.getPredictionFromClassifier(sess=sess,ind=i,test_in=data)
            for j in range(tt):
                aa[j][out[j]] = aa[j][out[j]]  + 1
        bb = [np.argmax(aa[i]) for i in range(tt)]
        return bb
