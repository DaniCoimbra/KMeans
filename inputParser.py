import csv
import os
import inputParser
import generator

def openFile(filePath):
    dataset=[]
    with open(filePath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            dataPoint=row[0].split(",")
            dataset.append((float(dataPoint[0]),float(dataPoint[1])))
    return dataset

def inputDataset(input):
    datapoints = input.split(" ")
    dataset=[]
    for datapoint in datapoints:
        dp = datapoint.split(",")
        dataset.append((float(dp[0]),float(dp[1])))
    return dataset

def parse(rawInput):
    if os.path.isfile(rawInput):
        dataset = inputParser.openFile(rawInput)
    elif rawInput.isdigit():
        dataset = generator.generateDataset(int(rawInput))
    else:
        dataset = inputParser.inputDataset(rawInput)
    return dataset