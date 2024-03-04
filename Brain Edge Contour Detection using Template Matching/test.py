import glob
import os
import cv2
import numpy as np
from PIL import Image

def templateMatch(file,template):
    temp=cv2.imread(file)
    img=cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    #template=img[135:143, 236:242]
    w,h=template.shape[::-1]    
    result=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    thresh_val=0.95
    ver=np.where(result>=thresh_val)[::-1]
    return w,h,ver,temp,img
    
def contourDetect(file):
    img=cv2.imread(file)
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurGrayImg=cv2.GaussianBlur(grayImg,(3,3),0)
    estThreshVal,threshImg=cv2.threshold(blurGrayImg,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    contours,hierarchy=cv2.findContours(threshImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    imgCopy=np.copy(img)
    for i, c in enumerate(contours):
        cv2.drawContours(imgCopy,contours,i,(0,255,0),3)        
    return imgCopy,contours

def cv2ToPil(img_cv2):
    img_array=cv2.cvtColor(img_cv2,cv2.COLOR_BGR2RGB)
    img_pil=Image.fromarray(img_array)
    return img_pil

def pilToCV2(img_pil):
    img_pil=img_pil.convert('RGB')
    img_array=np.array(img_pil)
    img_cv2=img_array[:,:,::-1]
    return img_cv2

path=os.getcwd()
#print(path)

path_dataset=glob.glob(str(path)+'/testPatient/*_thresh.png')

path_slice= str(path)+'/Slices/'
if not os.path.exists(path_slice):
    os.mkdir(path_slice)
else:
    print('Slice Directory already present!!!')

path_boundary=str(path)+'/Boundaries/'
if not os.path.exists(path_boundary):
    os.mkdir(path_boundary)
else:
    print('Boundary Directory already present!!!')


template=cv2.imread(str(path)+'/template_matcher.png',0)
    
a=1           #cnt for slice num
b=1           #cnt for IC num

for file in path_dataset:
    wid,hei,dim,img_org,img_gray=templateMatch(file,template)
    img_cd,contours=contourDetect(file)
    for vtx in zip(*dim):
        #print(vtx)
        
        slice_sub_path=str(path_slice)+'IC_'+str(a)+'_thresh/'
        if not os.path.exists(slice_sub_path):
            os.mkdir(slice_sub_path)
                
        img=cv2ToPil(img_org)
        img_slice=img.crop((vtx[0]+wid,vtx[1]+hei,vtx[0]+118,vtx[1]+118))
        pxl_num=np.sum(pilToCV2(img_slice)>0)         #Number of non-black pixels
        if(pxl_num==0):
            continue
        else:
            img_slice.save(str(slice_sub_path)+'Slice_'+str(b)+'.png')		
        b+=1
    b=1    
    a+=1

c=1        #cnt for boundary num
d=1        #cnt for IC num

path_slice_dir=glob.glob(str(path_slice)+'*')

for f1 in path_slice_dir:
    path_slice_sub=glob.glob(str(f1)+'/*')
    for f2 in path_slice_sub:
        bndry_sub_path=str(path_boundary)+'IC_'+str(d)+'_thresh/'
        if not os.path.exists(bndry_sub_path):
            os.mkdir(bndry_sub_path)
        img_org_cd,ctr=contourDetect(f2)
        loc=str(bndry_sub_path)+'Boundary_'+str(c)+'.png'
        cv2.imwrite(str(loc),img_org_cd)
        c+=1
    c=1
    d+=1
