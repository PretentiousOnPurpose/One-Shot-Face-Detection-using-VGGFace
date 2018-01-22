from PIL import Image
import numpy as np

imgs = np.load("imgs.npy")
labels = np.load("labels.npy")

def load_data1():
    data = []
    for img in imgs[:500]:
        data.append(np.array(Image.open(img[:-4] + "jpg")))

    data = np.array(data)
    batch = []
    for i in range(500):
        batch.append(np.array([ data[i]/255, labels[:500][i] ]))

    return np.array(batch)

def load_data2():
    data = []
    for img in imgs[500:1000]:
        data.append(np.array(Image.open(img[:-4] + "jpg")))

    data = np.array(data)
    batch = []
    for i in range(500):
        batch.append(np.array([ data[i]/255, labels[500:1000][i] ]))

    return np.array(batch)

def load_data3():
    data = []
    for img in imgs[1000:1500]:
        data.append(np.array(Image.open(img[:-4] + "jpg")))

    data = np.array(data)
    batch = []
    for i in range(500):
        batch.append(np.array([ data[i]/255, labels[1000:1500][i] ]))

    return np.array(batch)

def load_data4():
    data = []
    for img in imgs[1500:2000]:
        data.append(np.array(Image.open(img[:-4] + "jpg")))

    data = np.array(data)
    batch = []
    for i in range(500):
        batch.append(np.array([ data[i]/255, labels[1500:2000][i] ]))

    return np.array(batch)

def load_data5():
    data = []
    for img in imgs[2000:2500]:
        data.append(np.array(Image.open(img[:-4] + "jpg")))

    data = np.array(data)
    batch = []
    for i in range(500):
        batch.append(np.array([ data[i]/255, labels[2000:2500][i] ]))

    return np.array(batch)

def load_data6():
    data = []
    for img in imgs[2500:3000]:
        data.append(np.array(Image.open(img[:-4] + "jpg")))

    data = np.array(data)
    batch = []
    for i in range(500):
        batch.append(np.array([ data[i]/255, labels[2500:3000][i] ]))

    return np.array(batch)

def load_data7():
    data = []
    for img in imgs[3000:3500]:
        data.append(np.array(Image.open(img[:-4] + "jpg")))

    data = np.array(data)
    batch = []
    for i in range(500):
        batch.append(np.array([ data[i]/255, labels[3000:3500][i] ]))

    return np.array(batch)

def load_data8():
    data = []
    for img in imgs[3500:]:
        data.append(np.array(Image.open(img[:-4] + "jpg")))

    data = np.array(data)
    batch = []
    for i in range(500):
        batch.append(np.array([ data[i]/255, labels[3500:][i] ]))

    return np.array(batch)
