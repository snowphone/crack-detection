'''
epoch: 데이터 세트를 한번 순회함. 즉, 1 epoch는 1번 데이터 세트를 순회함을 의미.
batch_size: 가중치 갱신을 하기 까지 입력받을 데이터 수
iteration: epoch / batch_size

tf.nn.softmax_cross_entropy_with_logits는 deprecated이지만, _v2에 비해 성능이 좋으므로
그대로 유지한다.

'''
from time import time

import numpy as np
import tensorflow as tf

from misc import grade, label, load, load_file_name

EPOCH = 100
LEARNING_RATE = 0.001
path = "./images/"
#학습용 사진 및 라벨
x_images, y_labels = load(path + "train/")
#확인용 사진 및 라벨
test_images, test_labels = load(path + "test/")

def main():
	train()
	predict()
	print("정확도: {:.3f}%".format(100 * accuracy()))
	return

def test_algorithm():
	f = open("result.log", mode="w+")
	f.write("epoch: {}, learning rate: {}\n".format(EPOCH, LEARNING_RATE))
	accu_list = []
	for i in range(5):
		train()
		accu = accuracy()
		accu_list.append(accu)
		info = "{}th accuracy: {}\n".format(i+1, accu)
		print(info)
		f.write(info)
	f.write("5번의 평균 정확도: {}\n".format(sum(accu_list) / len(accu_list)))


def CNN_layer(input, filters):
	'''convolution layer 구축'''
	layer = tf.layers.conv2d(input, filters, [3, 3], activation=tf.nn.relu)
	layer = tf.layers.max_pooling2d(layer, [2, 2], [2, 2])
	layer = tf.layers.dropout(layer, 0.7, is_training)
	return layer


def train():
	init = tf.global_variables_initializer()
	begin = time()
	sess.run(init)

	for epoch in range(EPOCH):
		_, total_cost = sess.run([optimizer, cost],
								 feed_dict={X: x_images,
											Y: y_labels,
											is_training: True})

		print('Epoch:', '%04d' % (epoch + 1),
			  'Avg. cost =', '{:.3E}'.format(total_cost / len(x_images)))

	print('최적화 완료!')
	print("학습 데이터: {}\nepoch: {}".format(len(x_images), EPOCH))
	print("소요시간: {:.1f}초".format(time() - begin))
	return

def predict():
	'''테스트 데이터 셋에 대해 예측답 및 실제 값을 보여줌.'''
	file_names = load_file_name(path + "test/")
	model_idxes = sess.run(tf.argmax(model,1), feed_dict={X:test_images, is_training:False})
	answer_idxes = sess.run(tf.argmax(Y,1), feed_dict={Y:test_labels}) 
	print("이미지\t\t예측답\t실제답\t일치여부")
	print(*[(name, grade[predict], grade[target], predict == target) 
		for name, predict, target 
		in zip(file_names, model_idxes, answer_idxes)], sep='\n')
	return

def accuracy():
	is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
	return sess.run(accuracy,
					feed_dict={X: test_images,
							Y: test_labels,
							is_training: False})


if __name__ == "__main__":
	''' 신경망 모델 구성 및 학습'''
	# 250 * 250 pixel, rgb채널의 이미지를 가지는 배열
	X = tf.placeholder(tf.float32, [None, 250, 250, 3])
	# 5개의 등급으로 구분되는 one-hot-vector array
	Y = tf.placeholder(tf.float32, [None, 5])
	is_training = tf.placeholder(tf.bool)

	L1 = CNN_layer(X, 32)
	L2 = CNN_layer(L1, 64)
	L3 = CNN_layer(L2, 128)

	#평면화
	L4 = tf.contrib.layers.flatten(L3)
	#dnn
	L4 = tf.layers.dense(L4, 1024, activation=tf.nn.relu)
	L4 = tf.layers.dropout(L4, 0.5, is_training)
	L5 = tf.layers.dense(L4, 1024, activation=tf.nn.relu)
	L5 = tf.layers.dropout(L5, 0.5, is_training)

	model = tf.layers.dense(L5, len(grade), activation=None)

	cost = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

	sess = tf.Session()

	main()
