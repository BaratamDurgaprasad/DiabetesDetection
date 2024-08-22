from numpy import loadtxt
from keras.models import model_from_json

dataset = loadtxt('pima-indians-diabetes.csv',delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("model_weights.weights.h5")
print("Loaded model from disk")

predictions = model.predict(x)
predictions = (predictions >0.5).astype(int)
for i in range(5,10):
        print('%s => %d (Originl Class: %d)' % (x[i].tolist(),predictions[i][0], int(y[i])))
        

