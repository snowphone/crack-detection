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

from misc import label, load, grade

EPOCH = 15
LEARNING_RATE = 0.001


path = "./images/"
x_images, y_labels = load(path + "train/")
test_images, test_labels = load(path + "test/")

#########
# 신경망 모델 구성
######
# 250 * 250 pixel, rgb채널의 이미지를 가지는 배열
X = tf.placeholder(tf.float32, [None, 250, 250, 3])
# 5개의 등급으로 구분되는 one-hot-vector array
Y = tf.placeholder(tf.float32, [None, 5])
is_training = tf.placeholder(tf.bool)

L1 = tf.layers.conv2d(X, 32, [3, 3], activation=tf.nn.relu)
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
L1 = tf.layers.dropout(L1, 0.7, is_training)

L2 = tf.layers.conv2d(L1, 64, [3, 3], activation=tf.nn.relu)
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.7, is_training)

L3 = tf.contrib.layers.flatten(L2)
L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.5, is_training)

model = tf.layers.dense(L3, len(grade), activation=None)

cost = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
with tf.Session() as sess:
	begin = time()
	sess.run(init)

	for epoch in range(EPOCH):
		_, total_cost = sess.run([optimizer, cost],
								 feed_dict={X: x_images,
											Y: y_labels,
											is_training: True})

		print('Epoch:', '%04d' % (epoch + 1),
			  'Avg. cost =', '{:.4f}'.format(total_cost / len(x_images)))

	print('최적화 완료!')
	print("학습 데이터: {}\nepoch: {}".format(len(x_images), EPOCH))

	#########
	# 결과 확인
	######

	#테스트 데이터 셋에 대해 예측답 및 실제 값을 보여줌.
	model_idxes = sess.run(tf.argmax(model,1), feed_dict={X:test_images, is_training:False})
	answer_idxes = sess.run(tf.argmax(Y,1), feed_dict={Y:test_labels}) 
	print("예측답\t실제답")
	print(*[(grade[predict], grade[target]) for predict,target in zip(model_idxes, answer_idxes)], sep='\n')


	is_correct = tf.equal(tf.argmax(model,1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
	print("정확도:", sess.run(accuracy,
							feed_dict={X: test_images,
									Y: test_labels,
									is_training: False}))
	print("소요시간: {:.1f}초".format(time() - begin))
