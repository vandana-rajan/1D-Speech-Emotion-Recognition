# use raw time-domain speech signal as input to cnn for SER

import numpy as np
import os
import librosa


datapath = '/data/home/vandana/emodb/wav'
classes = ['W','L','E','A','F','T','N'] # 7 classes

seg_len = 16000 # signal split length (in samples) in time domain
seg_ov = int(seg_len*0.5) # 50% overlap

def normalize(s):
# RMS normalization
	new_s = s/np.sqrt(np.sum(np.square((np.abs(s))))/len(s))
	return new_s

def countclasses(fnames):
	dict = {classes[0]:0,classes[1]:0,classes[2]:0,classes[3]:0,classes[4]:0,classes[5]:0,classes[6]:0}
	for name in fnames:
		if name[5] in classes:
			dict[name[5]]+=1
	return dict

def data1d(path):

	fnames = os.listdir(datapath)
	dict = countclasses(fnames)
	print('Total Data',dict)
	num_cl = len(classes)
	train_dict = {classes[0]:0,classes[1]:0,classes[2]:0,classes[3]:0,classes[4]:0,classes[5]:0,classes[6]:0}
	test_dict = {classes[0]:0,classes[1]:0,classes[2]:0,classes[3]:0,classes[4]:0,classes[5]:0,classes[6]:0}
	val_dict = {classes[0]:0,classes[1]:0,classes[2]:0,classes[3]:0,classes[4]:0,classes[5]:0,classes[6]:0}

	for i in range(num_cl):
		cname =  dict.keys()[i]
		cnum = dict[cname]
		t = round(0.8*cnum)
		test_dict[cname] = int(cnum - t)
		val_dict[cname] = int(round(0.2*t))
		train_dict[cname] = int(t - val_dict[cname])
		print('Class:',cname,'train:',train_dict[cname],'val:',val_dict[cname],'test:',test_dict[cname])
		
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	x_val = []
	y_val = []

	count = {classes[0]:0,classes[1]:0,classes[2]:0,classes[3]:0,classes[4]:0,classes[5]:0,classes[6]:0}

	for name in fnames:
		if name[5] in classes:
			sig,fs = librosa.load(datapath+'/'+name, sr=16000)
			# normalize signal
			data = normalize(sig)
			if(len(data) < seg_len):
				pad_len = int(seg_len - len(data))
				pad_rem = int(pad_len % 2)
				pad_len /= 2
				signal = np.pad(data,(int(pad_len), int(pad_len+pad_rem)),'constant',constant_values=0)
			elif(len(data) > seg_len):
				signal = []
				end = seg_len
				st = 0
				while(end < len(data)):
					signal.append(data[st:end])
					st = st + seg_ov
					end = st + seg_len
				signal = np.array(signal)
				if(end >= len(data)):
					num_zeros = int(end-len(data))
					if(num_zeros > 0):
						n1 = np.array(data[st:end])
						n2 = np.zeros([num_zeros])
						s = np.concatenate([n1,n2],0)
					else:
						s = np.array(data[int(st):int(end)])
				signal = np.vstack([signal,s])
			else:
				signal = data

			if(count[name[5]] < train_dict[name[5]]):
				if(signal.ndim>1):
					for i in range(signal.shape[0]):						
						x_train.append(signal[i])
						y_train.append(name[5])
				else:
					x_train.append(signal)
					y_train.append(name[5])
			else:
				if((count[name[5]]-train_dict[name[5]]) < val_dict[name[5]]):
					if(signal.ndim>1):
						for i in range(signal.shape[0]):							
							x_val.append(signal[i])
							y_val.append(name[5])
					else:
						x_val.append(signal)
						y_val.append(name[5])
				else:
					if(signal.ndim>1):
						for i in range(signal.shape[0]):
							x_test.append(signal[i])
							y_test.append(name[5])
					else:
						x_test.append(signal)
						y_test.append(name[5])
			count[name[5]]+=1
	return np.float32(x_train),y_train,np.float32(x_test),y_test,np.float32(x_val),y_val

def string2num(y):
	y1 = []
	for i in y:
		if(i == classes[0]):
			y1.append(0)
		elif(i == classes[1]):
			y1.append(1)
		elif(i == classes[2]):
			y1.append(2)
		elif(i == classes[3]):
			y1.append(3)
		elif(i == classes[4]):
			y1.append(4)
		elif(i == classes[5]):
			y1.append(5)
		else:
			y1.append(6)
	y1 = np.float32(np.array(y1))
	return y1

def load_data():
	x_tr,y_tr,x_t,y_t,x_v,y_v = data1d(datapath)
	y_tr = string2num(y_tr)
	y_t = string2num(y_t)
	y_v = string2num(y_v)
	return x_tr, y_tr, x_t, y_t, x_v, y_v
	
	
					



