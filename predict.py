import joblib

model = joblib.load('E:\iris_classifier\models\model.pkl')
print("model loaded sucessfully")
print(model.predict([[5.1, 3.5, 1.4, 0.2]]))