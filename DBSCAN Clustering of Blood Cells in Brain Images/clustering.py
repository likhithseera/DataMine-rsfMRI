import glob
import os
import cv2
import csv
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn import metrics

def templateMatch(file,template):
    temp=cv2.imread(file)
    norm_img=np.zeros((800,800))
    norm_temp=cv2.normalize(temp,norm_img,0,255,cv2.NORM_MINMAX)
    img=cv2.cvtColor(norm_temp, cv2.COLOR_BGR2GRAY)
    #template=img[135:143, 236:242]
    #template=cv2.imread('C:/Users/likhi/OneDrive/Desktop/template_matcher.png',0)
    w,h=template.shape[::-1]    
    result=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    thresh_val=0.95
    ver=np.where(result>=thresh_val)[::-1]
    return w,h,ver,temp,img
    
def clusterDetect(file):
    img=cv2.imread(file)
    hsvImg=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsvImg,(0,0,0),(255,0,255))
    nzmask=cv2.inRange(hsvImg,(0,0,0),(255,255,255))
    nzmask=cv2.erode(nzmask,np.ones((3,3)))
    mask=mask&nzmask
    result=img.copy()
    result[np.where(mask)]=0
    #result[mask==0]=(0,255,255)
    return result

def findClusterUsingDBSCAN(a):
    if(len(a.shape)<3):
        Z=a.reshape((-1,1))
    elif len(a.shape)==3:
        Z=a.reshape((-1,3))
    Z=np.float32(Z[:,:2])
    dbs=DBSCAN(eps=1, min_samples=10).fit(Z)
    img_pixels,_=Z.shape
    core_samples_mask=np.zeros_like(dbs.labels_,dtype=bool)
    core_samples_mask[dbs.core_sample_indices_]=True
    labels=dbs.labels_
    non_black_pixels=cv2.countNonZero(Z)
    black_pixels=img_pixels-non_black_pixels
    clusters_num=len(set(labels))-(1 if -1 in labels else 0)
    print(non_black_pixels,black_pixels)
    if(non_black_pixels>135):
        for i in set(labels):
            if(i==-1):
                continue
            elif(i==0):
                cluster_pxl_num=np.sum(labels==i)-black_pixels
                if(cluster_pxl_num<=135):
                    clusters_num-=1
                    #print(i,cluster_pxl_num,clusters_num)
                else:
                    continue
            else:
                cluster_pxl_num=np.sum(labels==i)
                if(cluster_pxl_num<=135):
                    clusters_num-=1
                    #print(i,cluster_pxl_num,clusters_num)
                else:
                    continue
                    #print(i,cluster_pxl_num,clusters_num)
    else:
        clusters_num=0          
    return clusters_num,dbs.eps,dbs.min_samples
    
def cv2ToPil(img_cv2):
    img_array=cv2.cvtColor(img_cv2,cv2.COLOR_BGR2RGB)
    img_pil=Image.fromarray(img_array)
    #img_pil.save('C:/Users/likhi/OneDrive/Desktop/template_matcher3.png')
    return img_pil

def pilToCV2(img_pil):
    img_pil=img_pil.convert('RGB')
    img_array=np.array(img_pil)
    img_cv2=img_array[:,:,::-1]
    #img_cv2.save('C:/Users/likhi/OneDrive/Desktop/template_matcher4.png')
    return img_cv2

    
path='C:/Users/likhi/OneDrive/Desktop'
#path=os.getcwd()
print(path)

path_dataset=glob.glob(str(path)+'/testPatient/*_thresh.png')

path_slice= str(path)+'/Slices/'
if not os.path.exists(path_slice):
    os.mkdir(path_slice)
else:
    print('Slice Directory already present!!!')

path_clusters=str(path)+'/Clusters/'
if not os.path.exists(path_clusters):
    os.mkdir(path_clusters)
else:
    print('Cluster Directory already present!!!')

template=cv2.imread(str(path)+'/template_matcher.png',0)
    
a=1           #cnt for slice num
b=1           #cnt for IC num

for file in path_dataset:
    wid,hei,dim,img_org,img_gray=templateMatch(file,template)
    #img_cd,contours=contourDetect(file)
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

e=1        #cnt for cluster num
f=1        #cnt for IC num
data={}
path_slice_dir=glob.glob(str(path_slice)+'*')
for f1 in path_slice_dir:
    path_slice_sub=glob.glob(str(f1)+'/*') 
    for f2 in path_slice_sub:
        clstr_sub_path=str(path_clusters)+'IC_'+str(f)+'_thresh/'
        if not os.path.exists(clstr_sub_path):
            os.mkdir(clstr_sub_path)
        img_org_cd=clusterDetect(f2)
        n_clusters,eps,min_pts=findClusterUsingDBSCAN(img_org_cd)
        loc=str(clstr_sub_path)+'Cluster_'+str(e)+'.png'
        data[e]=n_clusters
        if(n_clusters>=0):
            cv2.imwrite(str(loc),img_org_cd)
        e+=1
    e=1
    print('Epsilon: '+str(eps)+" MinPoints: "+str(min_pts))
    fl=str(path_clusters)+'IC_'+str(f)+'_thresh/IC_'+str(f)+'_thresh.csv'
    with open(fl,'w',newline='') as csvfile:
              header_key=['SliceNumber','ClusterCount']
              csvWrite=csv.DictWriter(csvfile,fieldnames=header_key)
              csvWrite.writeheader()
              for i in data:
                  csvWrite.writerow({'SliceNumber':i,'ClusterCount':data[i]})
    print('CSV Written for IC_'+str(f)+'_thresh.csv')
    data.clear()
    f+=1
