
# 1. 데이터
import numpy as np

x = np.array(range(1, 101)) # 테스트 하고싶은 데이터
y = np.array(range(101, 201))

x_train = x[:70]
y_train = y[:70]
x_test = x[:30]
y_test = y[:30]

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
            metrics=['mse'])

model.fit(x_train, y_train, epochs=100, validation_split=0.2) 
            #validation_data=(x_val, y_val))

# 4. 평가, 예측
# loss, acc = model.evaluate(x, y)
loss = model.evaluate(x_test, y_test)

print("loss : ", loss)
# print("acc : ", acc)

# 4. 예측
y_predict = model.predict(x_test)
print("결과물 : \n : ", y_predict)

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


