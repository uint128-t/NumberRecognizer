import math
import config
import random
def aget(arr,i,j):
    if i<0 or i>=28 or j<0 or j>=28:
        return 0
    return arr[j*28+i]
def ptrotate(x,y,angle):
    st=math.sin(angle)
    ct=math.cos(angle)
    return x*ct-y*st, x*st+y*ct
def imagetransform(image:list[int],scale:float,tx:float,ty:float,rotation:float):
    out=[0.0]*28*28
    for x in range(28):
        for y in range(28):
            nx=(x-tx)/scale
            ny=(y-ty)/scale
            nx,ny=ptrotate(nx-4,ny-4,rotation)
            nx+=4
            ny+=4
            x1=math.floor(nx)
            x2=x1+1
            y1=math.floor(ny)
            y2=y1+1
            out[y*28+x]+=aget(image,x1,y1)*(x2-nx)*(y2-ny)
            out[y*28+x]+=aget(image,x2,y1)*(nx-x1)*(y2-ny)
            out[y*28+x]+=aget(image,x1,y2)*(x2-nx)*(ny-y1)
            out[y*28+x]+=aget(image,x2,y2)*(nx-x1)*(ny-y1)
    return out

if __name__ == "__main__":
    rec=open(config.RECOGNIZE,"rb")
    img=list(rec.read(28*28))
    scale=random.uniform(0.7,1.3)
    corner=4*scale
    newsz=20*scale
    maxshift=28-newsz
    trf=imagetransform(img,scale,maxshift-corner,maxshift-corner,0.2)
    print(maxshift)
    config.print_drawing(trf)