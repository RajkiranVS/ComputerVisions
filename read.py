import csv
import numpy as np

filename = '/home/faceopen/Downloads/27352_34877_compressed_mnist_test.csv/mnist_test.csv'

with open(filename) as training_file:
    reader = csv.reader(training_file)
    next(reader, None)
    labels = []
    images = []
    for row in reader:
        label= row[0]
        labels.append(label)
        image= row[1:785]
        images.append(image)
    print(len(labels))