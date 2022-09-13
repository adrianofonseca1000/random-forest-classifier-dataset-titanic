from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifierModel(object):

    def model(self, X_train, Y_train, X_test, Y_test):
        model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42).fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        if self.model is None:
            print("Modelo não carregado.")
            return
        print("Métricas Training:")
        print("Accuracy:", accuracy_score(Y_test, Y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(Y_test, Y_pred))
        print("Classification report:")
        print(classification_report(Y_test, Y_pred))

        return model