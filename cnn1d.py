# 1D cnn for SER

from keras.models import Model,Sequential
from keras import optimizers
from keras.layers import Input,Conv1D,BatchNormalization,MaxPooling1D,LSTM,Dense,Activation,Layer
from emodata1d import load_data
from keras.utils import to_categorical
import keras.backend as K
import argparse
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

def emo1d(input_shape, num_classes,args):
	
	model = Sequential(name='Emo1D')
	
	# LFLB1
	model.add(Conv1D(filters = 64,kernel_size = (3),strides=1,padding='same',data_format='channels_last',input_shape=input_shape))	
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size = 4, strides = 4))

	#LFLB2
	model.add(Conv1D(filters=64, kernel_size = 3, strides=1,padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size = 4, strides = 4))

	#LFLB3
	model.add(Conv1D(filters=128, kernel_size = 3, strides=1,padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size = 4, strides = 4))

	#LFLB4
	model.add(Conv1D(filters=128, kernel_size = 3, strides=1,padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(MaxPooling1D(pool_size = 4, strides = 4))

	#LSTM
	model.add(LSTM(units=args.num_fc)) 
		
	#FC
	model.add(Dense(units=num_classes,activation='softmax'))

	#Model compilation	
	opt = optimizers.SGD(lr = args.learning_rate, decay=args.decay, momentum=args.momentum, nesterov=True)
	model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['categorical_accuracy'])
	
	return model


def train(model,x_tr,y_tr,x_val,y_val,args):
	
	es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
	mc = ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', mode='max', verbose=1, save_best_only=True)
	history = model.fit(x_tr,y_tr,epochs=args.num_epochs,batch_size=args.batch_size,validation_data=(x_val,y_val),callbacks=[es, mc])
	return model

def test(model,x_t,y_t):

	saved_model = load_model('best_model.h5')
	score = saved_model.evaluate(x_t,y_t,batch_size=20)
	print(score)
	return score

def loadData():

	x_tr,y_tr,x_t,y_t,x_val,y_val = load_data()
	x_tr = x_tr.reshape(-1,x_tr.shape[1],1)
	x_t = x_t.reshape(-1,x_t.shape[1],1)
	x_val = x_val.reshape(-1,x_val.shape[1],1)
	y_tr = to_categorical(y_tr)
	y_t = to_categorical(y_t)
	y_val = to_categorical(y_val)
	return x_tr,y_tr,x_t,y_t,x_val,y_val

if __name__ == "__main__":

	import numpy as np
	import matplotlib.pyplot as plt
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	
	#load data	
	x_tr,y_tr,x_t,y_t,x_val,y_val = loadData()	
	
	args.num_fc = 64
	args.batch_size = 32
	args.num_epochs = 1500 #best model will be saved before number of epochs reach this value
	args.learning_rate = 0.0001
	args.decay = 1e-6
	args.momentum = 0.9

	#define model
	model = emo1d(input_shape=x_tr.shape[1:],num_classes=len(np.unique(np.argmax(y_tr, 1))),args=args)
	model.summary()

	#train model
	model = train(model,x_tr,y_tr,x_val,y_val,args=args)
	
	#test model
	score = test(model,x_t,y_t)
