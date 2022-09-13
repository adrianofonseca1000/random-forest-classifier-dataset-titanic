# Internal libs
from br.Data import Data
from br.RandomForestModel import RandomForestClassifierModel


if __name__ == '__main__':
    data = Data(data=None)
    data.gen_features()

    print("Running Models")
    rf_clf = RandomForestClassifierModel()
    rf_clf.model(data.X_train, data.Y_train, data.X_test, data.Y_test)