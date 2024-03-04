import os
import re
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics

def templateMatch(file,template):
    temp=cv2.imread(file)
    norm_img=np.zeros((800,800))
    norm_temp=cv2.normalize(temp,norm_img,0,255,cv2.NORM_MINMAX)
    img=cv2.cvtColor(norm_temp, cv2.COLOR_BGR2GRAY)
    w,h=template.shape[::-1]    
    result=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
    thresh_val=0.95
    ver=np.where(result>=thresh_val)[::-1]
    return w,h,ver,temp,img
        
def cv2ToPil(img_cv2):
    img_array=cv2.cvtColor(img_cv2,cv2.COLOR_BGR2RGB)
    img_pil=Image.fromarray(img_array)
    return img_pil

def pilToCV2(img_pil):
    img_pil=img_pil.convert('RGB')
    img_array=np.array(img_pil)
    img_cv2=img_array[:,:,::-1]
    return img_cv2


path='C:/Users/likhi/OneDrive/Desktop/PatientData/'
template=cv2.imread('C:/Users/likhi/OneDrive/Desktop/PatientData/template_matcher.png',0)
i=1
j=1
temp=pd.DataFrame()
for dirname, _, filenames in os.walk(path):
    #print(dirname)
    for filename in filenames:
        df=pd.DataFrame()
        #match2=re.search("\_thresh.png$", filename)
        if(re.search("\_Labels.csv$", filename)):
            im=[]
            csv_path=os.path.join(dirname, filename)
            #print(csv_path)
            a=pd.read_csv(csv_path)
            df['IC']=a['IC']
            df['Label']=a['Label'].apply(lambda x : 1 if(x>0) else 0)
            if temp.empty:
                temp['IC']=a['IC']
                temp['Label']=a['Label'].apply(lambda x : 1 if(x>0) else 0)
                for k in range(1,len(a)+1):
                    loc=str(dirname)+'Patient_'+str(j)+'/IC_'+str(k)+'_thresh.png'
                    wid,hei,dim,img_org,img_gray=templateMatch(loc,template)
                    for vtx in zip(*dim):
                        img=cv2ToPil(img_org)
                        img_slice=img.crop((vtx[0],vtx[1],vtx[0]+944,vtx[1]+708))
                        img_req=pilToCV2(img_slice)
                        break
                    img_100x100=cv2.resize(img_req,(224, 224))
                    img_rgb=cv2.cvtColor(img_100x100,cv2.COLOR_BGR2RGB)
                    img_arr=img_rgb.flatten()
                    im.append(img_arr)
                temp['Image']=im
                j=j+1
            else:
                tempDf = pd.DataFrame(columns=['IC','Label','Image'])
                tempDf['IC'] = a['IC']
                tempDf['Label'] = a['Label'].apply(lambda x : 1 if(x>0) else 0)
                for k in range(1,len(a)+1):
                    loc=str(dirname)+'Patient_'+str(j)+'/IC_'+str(k)+'_thresh.png'
                    wid,hei,dim,img_org,img_gray=templateMatch(loc,template)
                    for vtx in zip(*dim):
                        img=cv2ToPil(img_org)
                        img_slice=img.crop((vtx[0],vtx[1],vtx[0]+944,vtx[1]+708))
                        img_req=pilToCV2(img_slice)
                        break
                    img_100x100=cv2.resize(img_req,(224, 224))
                    img_rgb=cv2.cvtColor(img_100x100,cv2.COLOR_BGR2RGB)
                    img_arr=img_rgb.flatten()
                    im.append(img_arr)
                tempDf['Image']=im
                temp=pd.concat([temp,tempDf])
                j=j+1    
#print(temp)
shuffled = shuffle(temp,random_state=1)
n_classes = temp['Label'].unique()
# print('Total number of unique categories:', n_classes)
num_class = temp['Label'].nunique()
# print('Total number of unique categories:', num_class)

img_lst=[]
for arr in temp['Image']:
    img_lst.append(arr)
X=np.array(img_lst)
X=X.astype('float32')
X=X/255.
# print(X)
# print(X.shape)
# print(type(X))
Y=temp['Label']
#print(Y.shape)

df_len=len(temp)
#print(df_len)
tst_split=0.3
trn_split=1-tst_split
#test_lim=round(df_len*tst_split)
train_lim=round(df_len*trn_split)
#print(test_lim,train_lim)
X_train_val=X[:train_lim,]
Y_train_val=Y[:train_lim,]
X_test=X[train_lim:,]
Y_test=Y[train_lim:,]
#print(X_train_val.shape)
#Reshaping for 3-D for CNN
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 1)
X_train_val = X_train_val.reshape(X_train_val.shape[0], img_rows, img_cols, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
# X = X.reshape(X.shape[0], img_rows, img_cols, 3)
#print(X.shape)

# Y_one_hot = tf.keras.utils.to_categorical(Y,num_classes = num_class)
Y_train_val_one_hot = tf.keras.utils.to_categorical(Y_train_val,num_classes = num_class)
Y_test_one_hot = tf.keras.utils.to_categorical(Y_test,num_classes = num_class)
#print(Y_train_one_hot)
#print(type(Y))
# print(Y_one_hot.shape)
#print(Y_train_val_one_hot.shape)

X_train, X_val, Y_train_one_hot, Y_val_one_hot = train_test_split(X_train_val,Y_train_val_one_hot, test_size=0.2,random_state=18)

#print(X_train.shape, X_val.shape, Y_train_one_hot.shape, Y_val_one_hot.shape)

model1 = tf.keras.models.Sequential()

model1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu',input_shape = (input_shape[0],input_shape[1],3)))
model1.add(tf.keras.layers.MaxPool2D(strides=(2,2)))
model1.add(tf.keras.layers.Dropout(0.1))

model1.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
model1.add(tf.keras.layers.MaxPool2D(strides=(2,2)))
model1.add(tf.keras.layers.Dropout(0.1))

model1.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))
model1.add(tf.keras.layers.MaxPool2D(strides=(2,2)))
model1.add(tf.keras.layers.Dropout(0.1))

model1.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), activation='relu'))
model1.add(tf.keras.layers.MaxPool2D(strides=(2,2)))
model1.add(tf.keras.layers.Dropout(0.1))

model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(256, activation='relu'))
model1.add(tf.keras.layers.Dropout(0.3))
model1.add(tf.keras.layers.Dense(num_class, activation='softmax'))
learning_rate = 0.0005

model1.compile(loss = 'binary_crossentropy',
              optimizer = tf.keras.optimizers.Adam(learning_rate),
              metrics=['accuracy'])

save_at = str(path)+"model.hdf5"
save_best = tf.keras.callbacks.ModelCheckpoint (save_at, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max')

history1 = model1.fit( X_train, Y_train_one_hot, 
                    epochs = 20, batch_size=1, 
                    callbacks=[save_best], verbose=1, 
                    validation_data = (X_val, Y_val_one_hot))

model_best = tf.keras.models.load_model(save_at)
score = model_best.evaluate(X_test, Y_test_one_hot, verbose=1)
print(score)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')

Y_pred = np.round(model1.predict(X_test))
print(Y_pred)

pred_score = model_best.evaluate(X_test, Y_pred, verbose=1)
print(pred_score)
print('Accuracy over the pred set: \n ', round((pred_score[1]*100), 2), '%')

#print(list(Y_test))
label=[]
for i in range(len(Y_pred)):
    if(Y_pred[i][0]==1.0):
        label.append(0)
    elif(Y_pred[i][1]==1.0):
        label.append(1)
    elif(Y_pred[i][0]==0.0):
        continue
    else:
        continue
#print(label)

confusion = metrics.confusion_matrix(list(Y_test), label)
print(confusion)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
total=(TP+TN+FP+FN)

accuracy=(TP+TN)/total
misclassification_error=(FP+FN)/total
precision=TP/(TP+FP)
recall=TP/(TP+FN)
specificity=TN/(TN+FP)
sensitivity=TP/(TP+FN)
f1_score=(2*TP)/((2*TP)+FP+FN)

data={
        'Accuracy':"{:.0%}".format(accuracy),
        'Misclassification-Error':"{:.0%}".format(misclassification_error),
        'Precision':"{:.0%}".format(precision),
        'Recall':"{:.0%}".format(recall),
        'F1-Score':"{:.0%}".format(f1_score),
        'Specificity':"{:.0%}".format(specificity),
        'Sensitivity':"{:.0%}".format(sensitivity)
    }
#print(data)