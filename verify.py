from config import *
from neuralnetwork import Layer
import struct
import neuralnetwork
import math
images=open(TEST_IMAGES,"rb")
labels=open(TEST_LABELS,"rb")
weights=open(WEIGHTS,"rb+")
# magic number
assert int.from_bytes(images.read(4))==2051
assert int.from_bytes(labels.read(4))==2049
N=int.from_bytes(images.read(4))
assert N==int.from_bytes(labels.read(4))
ROWS=int.from_bytes(images.read(4))
COLS=int.from_bytes(images.read(4))
LAYERSZ=ROWS*COLS
OUTSZ=10
print(f"{N} {ROWS}x{COLS} images")
def next_data():
    ans=int.from_bytes(labels.read(1))
    pict=list(images.read(LAYERSZ))
    return pict,ans

def nextweight():
    wt=weights.read(8)
    if len(wt)==8:
        return struct.unpack('d',wt)[0]
    else:
        raise ValueError("Not enough weights in file")
neuralnetwork.nextweight=nextweight

layers:list[Layer]=[]
layers.append(Layer(LAYERSZ,0)) # Input
pv=LAYERSZ
for sz in HIDDEN_SIZES:
    layers.append(Layer(sz,pv))
    pv=sz
layers.append(Layer(OUTSZ,pv))

correct=0

def train():
    global correct
    layerv,ans=next_data()
    layers[0].outputs=[x/255 for x in layerv]
    for li in range(1,len(layers)):
        layers[li].compute(layers[li-1])
    loss=0
    maxv=max(layers[-1].sums)
    maxi=layers[-1].sums.index(maxv)
    correct+=maxi==ans
    return loss,maxi==ans,maxv

for t in range(N):
    try:
        loss,cor,maxv=train()
        print(f"({t+1}/{N}) Correct:{'NY'[cor]} Accuracy: {100*correct/(t+1):.1f}%")
    except KeyboardInterrupt:
        print("Quit")
        break

weights.close()
print("Done")