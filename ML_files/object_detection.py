import os
import warnings
import itertools
from tensorflow.keras.utils import image_utils as image
from keras.src.utils import np_utils
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

train_dir= '../images/animals'
val_dir= '../images/val_dir'

train_class_labels=os.listdir(train_dir)
# print(train_class_labels)

train_total=0
for label in train_class_labels:
    total=len(os.listdir(os.path.join(train_dir,label)))
    # print(label,total)
    train_total+=total
# print('Train Total',train_total)

val_class_labels=os.listdir(val_dir)
val_total=0
for label in val_class_labels:
    total=len(os.listdir(os.path.join(val_dir,label)))
    # print(label,total)
    val_total+=total
# print('Val Total',val_total)

nb_train_samples=train_total
nb_val_samples=val_total
num_classes=88
img_rows=128
img_cols=128
channel=3

# Preprocess the train data
x_train=[]
y_train=[]
i=0
j=0
for label in train_class_labels:
    image_names_train=os.listdir(os.path.join(train_dir,label))
    total=len(image_names_train)
    # print("Train_Label", label)
    # print("Train_Total", total)
    for image_name in image_names_train:
        try:
            img=image.load_img(os.path.join(train_dir,label,image_name),target_size=(img_rows,img_rows,channel))
            img=image.img_to_array(img)
            img=img/255
            x_train.append(img)
            y_train.append(j)
        except:
            pass
        i+=1
    j+=1
x_train=np.array(x_train)
y_train=np.array(y_train)
y_train=np_utils.to_categorical(y_train[:nb_train_samples],num_classes)

# Preprocess the test data
x_test=[]
y_test=[]
i=0
j=0
for label in val_class_labels:
    image_names_test=os.listdir(os.path.join(val_dir,label))
    total=len(image_names_test)
    # print("Test_Label",label)
    # print("Test_Total",total)
    for image_name in image_names_test:
        try:
            img=image.load_img(os.path.join(val_dir,label,image_name),target_size=(img_rows,img_rows,channel))
            img=image.img_to_array(img)
            img=img/255.0
            x_test.append(img)
            y_test.append(j)
        except:
            pass
        i+=1
    j+=1
x_test=np.array(x_test)
y_test=np.array(y_test)
y_test=np_utils.to_categorical(y_test[:nb_val_samples],num_classes)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

# Model Building
model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(1026,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001,weight_decay=1e-04),
    metrics=['accuracy']
)

model.summary()

model.fit(
    x_train,
    y_train,
    batch_size=10,
    epochs=20,
    validation_data=(x_test,y_test),
    shuffle=True
)

y_pred=model.predict(x_test,batch_size=1,verbose=0)

y_predict=[]
for i in range(0,len(y_pred)):
    y_predict.append((int(np.argmax(y_pred[i]))))
len(y_predict)

y_true=[]
for i in range(0,len(y_test)):
    y_true.append(int(np.argmax(y_test[i])))
len(y_true)

def plot_confusion_matrix(cm,classes,title="Confusion Matrix",cmap=plt.cm.Blues):
    plt.figure(figsize=(10,10))
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)

    fmt='d'
    thresh=cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),horizontalalignment="center",color="white" if cm[i,j]>thresh else "Black")

    plt.ylabel('Actual Lable')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

cm_plot_labels=val_class_labels

print(classification_report(y_true=y_true,y_pred=y_predict))

# To plot confusion matrix
cm=confusion_matrix(y_true=y_true,y_pred=y_predict)
plot_confusion_matrix(cm,cm_plot_labels,title="Confusion Matrix")

score=model.evaluate(x=x_test,y=y_test,batch_size=32)
print('Test Accuracy', score[1])

score=model.evaluate(x=x_train,y=y_train,batch_size=32)
print('Train Accuracy', score[1])

predict_images=model.predict(x_train)

# model_structure=model.to_json()
# f=Path('Animal_Detection/model/model_structure.json')
# f.write_text(model_structure)
model.save_weights('../model/model_weights.h5')
model.save('../model/full_model.h5')



