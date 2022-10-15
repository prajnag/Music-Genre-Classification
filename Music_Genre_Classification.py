from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import random
import operator
import math

def get_neighbour(train_data, instance, k):
    distances = []
    for x in range(len(train_data)):
        dist = calc_distance(train_data[x], instance, k) + calc_distance(instance,train_data[x],k)
        distances.append((train_data[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getnearestclass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1 
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]

def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    return correct / len(testSet)


directory = '/Users/prajnagirish/Downloads/Data/genres_original'
f = open("dataset.dat", "wb")
i = 0
for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory+"/"+folder):
        try:
            (rate, sig) = wav.read(directory+"/"+folder+"/"+file)
            mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)
        except Exception as e:
            print("Got an exception: ", e, 'in folder: ', folder, ' filename: ', file)
f.close()

dataset = []

def loadDataset(filename, split, trset, teset):
    with open('data.dat','rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trset.append(dataset[x])
        else:
            teset.append(dataset[x])

train_data = []
testSet = []
loadDataset('data.dat', 0.70, train_data, testSet)

def calc_distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

length = len(testSet)
predictions = []
for x in range(length):
    predictions.append(getnearestclass(get_neighbour(train_data, testSet[x], 5)))

accuracy1 = getAccuracy(testSet, predictions)
print(accuracy1)