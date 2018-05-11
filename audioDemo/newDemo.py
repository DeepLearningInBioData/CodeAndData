#导入库
from __future__ import division,print_function,absolute_import
import tflearn
import speech_data
import tensorflow as tf
#定义参数
#learning rate是在更新权重的时候用，太高可用很快
#但是loss大，太低较准但是很慢
learning_rate=0.0001
training_iters=300000#STEPS
batch_size=64

width=20 #mfcc features
height=80 #(max) length of utterance
classes = 10  #digits

#用speech_data.mfcc_batch_generator获取语音数据并处理成批次，
#然后创建training和testing数据
batch=word_batch=speech_data.mfcc_batch_generator(batch_size)
X,Y=next(batch)
trainX,trainY=X,Y
testX,testY=X,Y #overfit for now

#4.建立模型
#speech recognition 是个many to many的问题
#所以用Recurrent NN
#通常的RNN，它的输出结果是受整个网络的影响的
#而LSTM比RNN好的地方是，它能记住并且控制影响的点，
#所以这里我们用LSTM
#每一层到底需要多少个神经元是没有规定的，太少了的话预测效果不好
#太多了会overfitting,这里普遍取128
#为了减轻过拟合的影响，我们用dropout,它可以随机地关闭一些神经元，
#这样网络就被迫选择其他路径，进而生成想对generalized模型
#接下来建立一个fully connected的层
#它可以使前一层的所有节点都连接过来，输出10类
#因为数字是0-9，激活函数用softmax,它可以把数字变换成概率
#最后用个regression层来输出唯一的类别，用adam优化器来使
#cross entropy损失达到最小

#Network building
net=tflearn.input_data([None,width,height])
net=tflearn.lstm(net,128,dropout=0.8)
net=tflearn.fully_connected(net,classes,activation='softmax')
net=tflearn.regression(net,optimizer='adam',learning_rate=learning_rate,loss='categorical_crossentropy')


#5.训练模型并预测
#然后用tflearn.DNN函数来初始化一下模型，接下来就可以训练并预测，最好再保存训练好的模型
#Traing
### add this "fix" for tensorflow version erros
col=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES,x)

model=tflearn.DNN(net,tensorboard_verbose=0)

while 1:  #training_iters
    model.fit(trainX, trainY, n_epoch=10, validation_set=(testX,testY), show_metric=True, batch_size=batch_size)
    _y=model.predict(X)
model.save("tflearn.lstm.model")
print(_y)