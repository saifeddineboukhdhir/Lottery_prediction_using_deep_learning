# -*- coding: utf-8 -*-
import pandas
import numpy as np 
import copy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Model
from keras.layers import concatenate, Add
from sklearn.metrics import classification_report,confusion_matrix
from keras.utils import to_categorical 
import matplotlib.pyplot as plt 
from keras.layers import Input, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import AveragePooling2D
from keras.initializers import glorot_uniform 


def get_data():
    df = pandas.read_csv('Lottery_Powerball_Winning_Numbers__Beginning_2010.csv')
    wining_numbers_list= df["Winning Numbers"].tolist()[:382]
    #e cross presence matrix defined as the number of times every pair of numbers appeared together
    cross_presence_matrices=[np.zeros((95,95,1)) for i in range(len(wining_numbers_list))]
    #cross_presence_matrix=np.zeros((95,95))
#    j=len(wining_numbers_list)-1
#    wining_numbers=[ int(a) for a in wining_numbers_list[j].split()]
#    wining_numbers[-1]=wining_numbers[-1]+69
#    for number1  in wining_numbers:
#        for number2 in wining_numbers:
#            cross_presence_matrices[j][number1-1,number2-1,0]+=1
#            if number1!=number2:
#                cross_presence_matrices[j][number2-1,number1-1,0]+=1
                
    j=len(wining_numbers_list)-2
    while j>=0:
        cross_presence_matrices[j]=copy.copy(cross_presence_matrices[j+1])
        wining_numbers=[ int(a) for a in wining_numbers_list[j].split(" ")]
        wining_numbers[-1]=wining_numbers[-1]+69
        for number1 in wining_numbers:
            for number2 in wining_numbers: 
                cross_presence_matrices[j][number1-1,number2-1,0]+=1
                if number1!=number2:
                    cross_presence_matrices[j][number2-1,number1-1,0]+=1
                

        j-=1
    drawing_times=[] #we added the number of times each number was drawn during all past draws
    for element in cross_presence_matrices:
        drawing_times.append(np.diagonal(element[:,:,0]))
    drawing_times=np.array(drawing_times)    
    cross_presence_matrices=np.array(cross_presence_matrices)
    date=df["Draw Date"].tolist()[:382]
    date=[[int(a) for a in element.split("/")] for element in date]
    date=np.array(date)
    X_data=np.concatenate((date,drawing_times),axis=1)
    y_data=np.zeros((len(wining_numbers_list),95))
    for i  in range (len(wining_numbers_list)):
        wining_numbers=[int(a) for a in wining_numbers_list[i].split()[:-1]]
        wining_numbers.append(int(wining_numbers_list[i].split()[-1])+69)
        for j in wining_numbers:
            y_data[i,j-1]=j
    
    #y_data=np.array([[int(a) for a in element.split()] for element in wining_numbers_list])
    #y_data[:,-1]+=69
    data=np.concatenate((X_data,y_data),axis=1)
    split = train_test_split(data, cross_presence_matrices, test_size=0.20,shuffle=True)
    (trainAttrX, testAttrX, trainCrossX, testCrossX) = split
    X_train=trainAttrX[:,:-95]
    y_train=trainAttrX[:,-95:]/95
    X_test=testAttrX[:,:-95]
    y_test=testAttrX[:,-95:]/95
    #print( X_test[-1,:],'\n',y_test[-1,:] )
    return(X_train,y_train,X_test,y_test,trainCrossX,testCrossX)
    #print(X_data.shape, y_data.shape, X_train.shape,y_train.shape)    
def create_mlp_model(inputDim):
    model = Sequential()
    model.add(Dense(150, activation="relu", input_dim=inputDim))
    model.add(Dense(80, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="sigmoid"))
    model.add(Dropout(0.1))
    return(model)

def create_cnn_model(width, height, depth):
    model=Sequential()
    model.add(Conv2D(180,(3,3),input_shape=(width,height,depth), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3,3 )))
    model.add(Conv2D(80,(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(3,3 )))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(150,activation="relu"))
    model.add(Dense(128,activation="relu"))
    model.add(Dense(80,activation="elu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(40,activation="relu"))
    model.add(Dense(28,activation="softmax"))  
    return model 
# this part of code is taken from the following website 
#https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff
# from here     
def identity_block(X,f,filters,stage,block):
    conv_name_base='res'+str(stage)+block+'_branch'
    bn_name_base='bn'+str(stage)+block+'_branch'
    
    F1, F2 ,F3=filters
    X_shortcut=X
    
    X=Conv2D(filters=F1, kernel_size=(1,1),strides=(1,1), padding='valid',name= conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
    X=Activation('relu')(X) 

    X=Conv2D(filters=F2, kernel_size=(f,f),strides=(1,1), padding='same',name= conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X=Activation('relu')(X) 



    X=Conv2D(filters=F3, kernel_size=(1,1),strides=(1,1), padding='valid',name= conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3, name=bn_name_base+'2c')(X)
    
    X=Add()([X,X_shortcut])
    X=Activation("relu")(X)
    return(X)
def convolutional_block(X,f,filters,stage,block,s=2):
    conv_name_base='res'+str(stage)+block+'_branch'
    bn_name_base='bn'+str(stage)+block+'_branch'
    
    F1, F2 ,F3=filters
    X_shortcut=X
    
    X=Conv2D(filters=F1, kernel_size=(1,1),strides=(s,s), padding='valid',name= conv_name_base+'2a',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
    X=Activation('relu')(X) 

    X=Conv2D(filters=F2, kernel_size=(f,f),strides=(1,1), padding='same',name= conv_name_base+'2b',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X=Activation('relu')(X) 



    X=Conv2D(filters=F3, kernel_size=(1,1),strides=(1,1), padding='valid',name= conv_name_base+'2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X=BatchNormalization(axis=3, name=bn_name_base+'2c')(X)
    
    X_shortcut=Conv2D(filters=F3, kernel_size=(1,1),strides=(s,s), padding='valid',name= conv_name_base+'1',kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut=BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)
    X=Add()([X,X_shortcut])
    X=Activation("relu")(X)
    return(X)  
      
def ResNet(input_shape = (95, 95, 1), classes = 95):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(128, (5, 5), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model 
# until here                    
def training_testing(): 
    X_train,y_train,X_test,y_test,X_cross_train,X_cross_test=get_data() 
    
    
    # create the MLP and CNN models
    mlp = create_mlp_model(X_train.shape[1])
    #cnn = create_cnn_model(95, 95, 1)
    cnn=ResNet()
    
    # create the input to our final set of layers as the *output* of both
    # the MLP and CNN
    combinedInput = concatenate([mlp.output, cnn.output])
    
    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = Dense(100, activation="relu")(combinedInput)
    x = Dense(95, activation="sigmoid")(x)
    
    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input, outputting a single value (the
    # predicted price of the car)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)
    
    # compile the model using mean absolute percentage error as our loss,
    # implying that we seek to minimize the absolute percentage difference
    # between our price *predictions* and the *actual prices*
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=['accuracy'])
    
    model.summary()
    
    # train the model
    print("[INFO] training model...")
    model.fit(
    	[X_train, X_cross_train], y_train,
    	validation_data=([X_test, X_cross_test], y_test),
    	epochs=10, batch_size=5)  
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_question2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_question2.h5")
    print("Saved model to disk")              
#    y_pred_test = model.predict([ X_test, X_cross_test])
#    print(y_pred_test[:1,:],'\n',y_test[:1,:])
#    report = classification_report(y_true=y_test,y_pred=y_)
#    matrix = confusion_matrix(y_true=y_test,y_pred=y_pred_test)
#    print("Training Set:")
#    print("report ",report)
#    print("matrix", matrix)
#    plt.matshow(matrix)
#    plt.colorbar()
#    plt.xlabel("Real class")
#    plt.ylabel("Predicted class")            
                
training_testing()
#get_data()