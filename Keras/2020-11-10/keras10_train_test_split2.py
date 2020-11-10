
# 1. 데이터
import numpy as np

x = np.array(range(1, 101)) # 테스트 하고싶은 데이터
y = np.array(range(101, 201))

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, test_size=0.2)
# 디폴드일 경우 shuffle은 true

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2. 모델 구성
model = Sequential()
model.add(Dense(300, input_shape=(1, ))) # = 스칼라1개
model.add(Dense(500))
model.add(Dense(1200))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
            metrics=['mse'])

model.fit(x_train, y_train, epochs=100,  
            validation_data=(x_val, y_val))

# 4. 평가, 예측
# loss, acc = model.evaluate(x, y)
loss = model.evaluate(x_test, y_test)

print("loss : ", loss)


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

print(x_test.shape) # 스칼라가 20개 
print(x_train.shape)
print(x_val.shape)