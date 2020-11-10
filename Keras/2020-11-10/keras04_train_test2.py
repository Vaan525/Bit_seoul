import numpy as np

# 1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 테스트 하고싶은 훈련용 데이터
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15]) # 평가 하고싶은(답을 알고있는) 데이터
y_test = np.array([11,12,13,14,15]) 
# train데이터와 test데이터를 나누지 않는다면, 답만 외우는 
# 형식의 훈련만 진행된다. 새로운 데이터에 대한 예측이 안된다.

# train과 test의 비율은 7 : 3

x_pred = np.array([16,17,18,19,20]) # 예측하고 싶은 데이터


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim=1))
model.add(Dense(70))
model.add(Dense(200))
model.add(Dense(70))
model.add(Dense(20))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)

# 4. 평가, 예측
# loss, acc = model.evaluate(x, y)
loss, acc = model.evaluate(x_test, y_test, batch_size=2)

print("loss : ", loss)
print("acc : ", acc)

# 4. 예측
y_pred = model.predict(x_pred)
print("결과물 : \n : ", y_pred)





