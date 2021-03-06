import numpy as np

# 1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10]) # 테스트 하고싶은 데이터
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15]) # 평가 하고싶은 데이터
y_test = np.array([11,12,13,14,15]) 
x_pred = np.array([16,17,18]) # 예측하고 싶은 데이터


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

model.fit(x_train, y_train, epochs=100)

# 4. 평가, 예측
# loss, acc = model.evaluate(x, y)
loss = model.evaluate(x_test, y_test)

print("loss : ", loss)
# print("acc : ", acc)

# 4. 예측
y_predict = model.predict(x_test)
print("결과물 : \n : ", y_predict)
# 새로운 예측값을 만들어서 아래에서 비교
# 

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
# 사용자 정의로 작성된 RMSE 
# RMSE를 사용하면 오류 지표를 실제 값과 유사한 단위로 다시 변환
# 해석을 쉽게 가능하다. 
