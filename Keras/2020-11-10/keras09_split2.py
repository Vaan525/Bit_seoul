
# 1. 데이터
import numpy as np

x = np.array(range(1, 101)) # 테스트 하고싶은 데이터
y = np.array(range(101, 201))

x_train = x[:60]   # 60개
x_val = x[:20]   # 20개
x_test = x[:20]   # 20개

y_train = y[:60]
y_val = y[:20]
y_test = y[:20]
# [60:80] 61 ~ 80


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# 2. 모델 구성
model = Sequential()
model.add(Dense(500, input_dim=1))
model.add(Dense(600))
model.add(Dense(2400))
model.add(Dense(600))
model.add(Dense(500))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
            metrics=['mse'])

model.fit(x_test, y_test, epochs=100,  
            validation_data=(x_val, y_val))


# 4. 평가, 예측
# loss, acc = model.evaluate(x, y)
loss, mse = model.evaluate(x_test, y_test)


print("loss : ", loss)
print("acc : ", mse)


# 4. 예측
y_predict = model.predict(x_test)
print("결과물 : \n : ", y_predict)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))


# R2
from sklearn.metrics import r2_score
r2 = r2_score(x_test, y_predict)
print("R2 : ", r2)


