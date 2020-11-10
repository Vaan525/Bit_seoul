
# 1. 데이터
import numpy as np

x = np.array(range(1, 101)) # 테스트 하고싶은 데이터
y = np.array(range(101, 201))

x_train = x[:60]
y_train = y[:60]

x_val = x[:20]
y_val = y[:20]

x_test = x[:20]
y_test = y[:20]

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
            metrics=['acc'])

model.fit(x_train, y_train, epochs=100,  
            validation_data=(x_val, y_val))


# 4. 평가, 예측
# loss, acc = model.evaluate(x, y)
loss = model.evaluate(x_train, y_train)


print("loss : ", loss)
# print("acc : ", acc)


# 4. 예측
y_predict = model.predict(x_train)
print("결과물 : \n : ", y_predict)


# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_train, y_predict)
print("R2 : ", r2)


