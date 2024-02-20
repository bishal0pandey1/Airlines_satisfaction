import joblib


def predict(data):
    model=joblib.load('output_model/RandomForest.sav')
    return model.predict(data)