import numpy as np
import glob
import scipy.ndimage

def imfilternoise(img,slevel):
    imgff=np.fft.fft2(img)
    imgff=np.where(np.abs(imgff)<slevel,0, imgff)
    return np.real(np.fft.ifft2(imgff))

def im2blur(img0,gsrad):
    fltrad=5*gsrad
    imgsz=img0.shape
    PSF=fspecial(fltrad,gsrad)
    img1=np.vstack((np.flipud(img0),img0,np.flipud(img0)))
    img1=np.hstack((np.fliplr(img1),img1,np.fliplr(img1)))
    imgbl=imfilter(img1,PSF)
    return imgbl[imgsz[0]:2*imgsz[0]+1,imgsz[1]:2*imgsz[1]+1]

def imfilter(img1,PSF):
    return scipy.ndimage.correlate(img1, PSF, mode='constant').transpose()

def fspecial(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def im2compare(img1,img2,rblur,vrange):
    diffx1=np.hstack((np.diff(img1[:,0:1],n=1,axis=2),0.5*(img1[:,2:]-img1[:,:-3]),np.diff(img1[:,-2:],n=1,axis=2)))
    diffx1=np.hstack((np.diff(diffx1[:,0:1],n=1,axis=2),0.5*(diffx1[:,2:]-diffx1[:,:-3]),np.diff(diffx1[:,-2:],n=1,axis=2)))
    diffy1=np.vstack((np.diff(img1[0:1,:]),0.5*(img1[2:]-img1[:-3,:]),np.diff(img1[-2:,:])))
    diffy1=np.vstack((np.diff(diffy1[0:1,:]),0.5*(diffy1[2:]-diffy1[:-3,:]),np.diff(diffy1[-2:,:])))
    laplace1=im2blur(diffx1+diffy1,5)

    diffx2=np.hstack((np.diff(img2[:,0:1],n=1,axis=2),0.5*(img2[:,2:]-img2[:,:-3]),np.diff(img2[:,-2:],n=1,axis=2)))
    diffx2=np.hstack((np.diff(diffx2[:,0:1],n=1,axis=2),0.5*(diffx2[:,2:]-diffx2[:,:-3]),np.diff(diffx2[:,-2:],n=1,axis=2)))
    diffy2=np.vstack((np.diff(img2(0:1,:)),0.5*(img2[2:]-img2[:-3,:]),np.diff(img2[-2:,:])))
    diffy2=np.vstack((np.diff(diffy2(0:1,:)),0.5*(diffy2[2:]-diffy2[:-3,:]),np.diff(diffy2[-2:,:])))
    laplace2=im2blur(diffx2+diffy2,rblur)

    mx,my=np.meshgrid(np.linspace(-np.abs(vrange[0]),np.abs(vrange[0])),np.linspace(-np.abs(vrange[1]),np.abs(vrange[1])),indexing='xy')
    mscore=np.zeros(mx.shape)

if __name__ == '__main__':
    thesourcefolder="C:\\Users\\Jeremias\\Desktop\\recordedimages\\im20220503141645"
    for thebatchround in range(9):
        print(thebatchround)
        thisroundfolder=str(thebatchround).zfill(5)
        thearrays=glob.glob(thesourcefolder+"\\"+thisroundfolder+"\\cam1\\*.npy")
        thetrue=0
        tobecompared=1
        while(True):
            if np.array_equal(np.load(thearrays[thetrue]),np.load(thearrays[tobecompared])):
                del thearrays[tobecompared]
                if thetrue==len(thearrays):
                    break
                elif tobecompared==len(thearrays):
                    thetrue=thetrue+1
                    tobecompared=thetrue+1
            else:
                if tobecompared==len(thearrays)-1 and tobecompared==thetrue+1:
                    break
                else:
                    thetrue=thetrue+1
                    tobecompared=thetrue+1
        thetimeslist=[]
        theaveragearray=np.zeros((2048,2048),dtype=np.int32)
        numframes=len(thearrays)
        for theimage in thearrays:
            thetimeslist.append(float(os.path.basename(theimage)[:-4]))
        img0=0
        img0=np.load(thearrays[11])[1000:2000,1000:2000]
        img0=imfilternoise(img,1000)
        xg,yg=np.meshgrid(np.linspace(1,img0.shape[1]),np.linspace(1,img0.shape[0]),indexing='xy')
        vdisp=[]
        maxspd=10
        rblur=5
        for theframe in range(12,41):
            img0s=im2blur(img0,5)
            img1=np.load(thearrays[theframe])[1000:2000+1,1000:2000+1]
            img1=imfilternoise(img1,1000)
            img1s=im2blur(img1,5)
            vdispa=im2compare(img1s,img0s,1,rblur,np.ones((1,2))*maxspd)
