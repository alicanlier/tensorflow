# 텐서플로우 수입
import tensorflow as tf
import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# MNIST 손글씨 데이터 package 수입
mnist = tf.keras.datasets.mnist


# MNIST 4분할 데이터 불러오기
(X_train,Y_train), (X_test,Y_test) = mnist.load_data()

print('학습용 입력데이터 모양 : ', X_train.shape)
print('학습용 출력데이터 모양 : ', Y_train.shape)
print('평가용 입력데이터 모양 : ', X_test.shape)
print('평가용 출력데이터 모양 : ', Y_test.shape)


#이미지 데이터 원본 출력

# plt.imshow(X_train[0], cmap='gray')
# plt.show()
# plt.imshow(X_train[1], cmap='Blues')
# plt.show()
#  
# print('첫번째 학습용 데이터 입력값', X_train[0])
# print('첫번째 학습용 데이터 출력값', Y_train[0])

#이미지 데이터 [0,1] 스케일링
 
X_train = X_train/255.0
X_test = X_test/255.0
 
#스케일링 후 데이터 확인
# plt.imshow(X_train[0], cmap='Greens')
# plt.show()
# print('첫번째 학습용 데이터 입력값', X_train[0])


#인공신경망 구현
model = tf.keras.models.Sequential()
layers = tf.keras.layers
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
 
#드롭아웃에 대한 작업
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))
 
# 인공신경망 작업에 대한 결과값 요약해서 보기
model.summary()
 
#인공신경망 학습 환경 설정
model.compile(optimizer='adam',loss ='sparse_categorical_crossentropy', metrics=['accuracy'])
 
# 인공신경망 학습
model.fit(X_train, Y_train, epochs=5, verbose=1)
 
# 인공신경만 평가
#인공신경망 예측
 
# 모델 저장
model.save('Predict_Model.h5')
 
# 딥러닝에 필요한 케라스 함수 호출
from keras.models import load_model
from keras.utils import np_utils
 
# 필요 라이브러리 호출(PIL은 이미지파일 처리위함)
from PIL import Image
import numpy as np
 
 
# test.png 파일 열어서 L(256단계 흑백이미지)로 변환
img = Image.open("손글씨테스트/data_1.jpg").convert("L")
 
# 이미지를 784개 흑백 픽셀로 사이즈 변환
img = np.resize(img, (1, 784))
img = img.reshape(1,28,28)
# 데이터를 모델에 적용할 수 있도록 가공
test_data = ((np.array(img) / 255) - 1) * -1
 
# 모델 불러오기
model = load_model('Predict_Model.h5')
 
# 클래스 예측 함수에 가공된 테스트 데이터 넣어 결과 도출
#res = model.predict_classes(test_data)
# 2021/10/02 수정 - 오류시 아래 명령어로 대체 가능합니다.
#res =(model.predict(test_data) > 0.5).astype("int32")
res=np.argmax(model.predict(test_data), axis=-1)
 
print('이미지 예측값 :', res)
