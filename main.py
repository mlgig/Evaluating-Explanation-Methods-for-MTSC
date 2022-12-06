from load_data import load_data
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.linear_model import RidgeClassifierCV, LinearRegression





def main():
    data = load_data("CMJ")

    minirocket = MiniRocketMultivariate()


    parameters = minirocket.fit(data["X_train"])
    X_train_trans = minirocket.transform(data["X_train"], parameters)
    X_test_trans = minirocket.transform(data["X_test"],parameters)
    print("time to transform (sec): ")
    cls = RidgeClassifierCV()
    cls.fit(X_train_trans,data["y_train"])
    print("Time to transform + train (sec):")

    acc = cls.score(X_test_trans,data["y_test"])
    print("Time to train + test (sec):" "\taccuracy is:",acc)









"""
    model = MrSEQLClassifier(seql_mode="fs")
    model.fit(data["X_train"],data["y_train"])
    print(model.score(data["X_test"],data["y_test"]))
"""
if __name__ == "__main__" :
    main()