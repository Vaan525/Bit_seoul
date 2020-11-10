
import numpy as np

# 1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 테스트 하고싶은 데이터
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_val = np.array([11,12,13,14,15])  # 검증용 데이터
y_val = np.array([11,12,13,14,15]) 
#x_pred = np.array([16,17,18]) # 예측하고 싶은 데이터
x_test = np.array([16,17,18,19,20])
y_test = np.array([16,17,18,19,20])

# train, test, val = 6 : 2 : 2

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

model.fit(x_train, y_train, epochs=100, 
            validation_data=(x_val, y_val))

# 4. 평가, 예측
# loss, acc = model.evaluate(x, y)
loss, mse = model.evaluate(x_test, y_test, batch_size=1)

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
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


