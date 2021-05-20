#1D Speech Emotion Recognition

Speech Emotion Recognition using raw speech signals from the EmoDB database using 1D CNN-LSTM architecture as given in the following paper.

Zhao, Jianfeng, Xia Mao, and Lijiang Chen. "Speech emotion recognition using deep 1D & 2D CNN LSTM networks." Biomedical Signal Processing and Control 47 (2019): 312-323.

EmoDB database can be downloaded from the following website

http://www.emodb.bilderbar.info/download/

There are 7 emotional classes and the validation accuracy obtained is ~61% as mentioned in the paper.

Developed and tested on the following:

Python 2.7
keras 2.2.4
Librosa 0.6.2

Update: 07-05-2021

Modifed the cnn1d.py architecture with attention mechanism (cnn1d_attn.py), shows better performance in terms of accuracy (66%-70% with lr 0.01).

##################################

#For another computational paralinguistic task, verbal conflict intensity estimation, see the repo https://github.com/smartcameras/ConflictNET
