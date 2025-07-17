from collections import deque
from neuralnetwork import Layer
from config import *
import random
import struct
import neuralnetwork
import transform
import math
images=open(TRAINING_IMAGES,"rb")
labels=open(TRAINING_LABELS,"rb")
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
    scale=random.uniform(0.7,1.3)
    corner=4*scale
    newsz=20*scale
    maxshift=28-newsz
    trf=transform.imagetransform(pict,scale,random.uniform(0,maxshift)-corner,random.uniform(0,maxshift)-corner,random.uniform(-0.2,0.2))
    # print_drawing(trf)
    return trf,ans

w=True
def nextweight():
    global w
    wt=weights.read(8)
    if len(wt)==8:
        return struct.unpack('d',wt)[0]
    else:
        if w:
            print(f"Warning: Not enough weights")
        w=False
        return random.uniform(-0.1, 0.1)
neuralnetwork.nextweight=nextweight

layers:list[Layer]=[]
layers.append(Layer(LAYERSZ,0)) # Input
pv=LAYERSZ
for sz in HIDDEN_SIZES:
    layers.append(Layer(sz,pv))
    pv=sz
layers.append(Layer(OUTSZ,pv))

def save_weights():
    weights.truncate(0)
    weights.seek(0)
    for layer in layers:
        for weight in layer.weights:
            weights.write(struct.pack('d',weight))
        for bias in layer.biases:
            weights.write(struct.pack('d',bias))

cr=deque()
al=deque()

def train():
    layerv,ans=next_data()
    layers[0].outputs=[x/255 for x in layerv]
    for li in range(1,len(layers)):
        layers[li].compute(layers[li-1])
    loss=0
    expect=[0]*OUTSZ
    expect[ans]=1
    out=layers[-1].sums
    maxi=out.index(max(out))
    ev=[math.e**v for v in out]
    sev=sum(ev)
    softmax=[v/sev for v in ev]
    cr.append(maxi==ans)
    if len(cr)>100:
        cr.popleft()
    for i in range(OUTSZ):
        diff=softmax[i]-expect[i]
        layers[-1].errorterm[i]=diff
        loss-=expect[i]*math.log(softmax[i]+1e-12)
    for i in reversed(range(1,len(layers)-1)):
        layers[i].backprop(layers[i+1])
        layers[i].derv(layers[i-1])
    for layer in layers:
        layer.update(LEARNING_RATE)
    al.append(loss)
    if len(al)>100:
        al.popleft()
    return loss

for t in range(N):
    try:
        loss=train()
        print(end=f"\r({t+1}/{N}) Accuracy: {100*cr.count(True)//len(cr)}%, Avg Loss: {sum(al)/len(al):.4f} Loss: {loss:.4f}   ")
    except KeyboardInterrupt:
        print("\nQuit")
        break

save_weights()
weights.close()
print("Done")