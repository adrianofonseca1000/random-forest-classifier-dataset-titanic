# Importing the libraries
# Bibliotecas para criação e manipulação de DATAFRAMES e Algebra
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


class Data():
    """
    Classe responsável por tratar os dados a serem utilizados
    """
    def __init__(self, data):
        self.data = data
        self.test_size = 0.33
        self.seed = 42

    print("Loading Features...")

    def gen_features(self):
        """
        :rtype: object
        """

        # Loading dataset train
        print("Loading data Train...")
        """
        Retorna pandas dataframe
        """
        self.data = pd.read_csv('titanic_data.csv', encoding="latin-1", sep=",")

        self.data.Age.fillna(self.data['Age'].mean(), inplace=True)

        # ___categorical feature
        #print("Converting categorical feature")
        var_cat = self.data.select_dtypes('object')
        for col in var_cat:
            self.data[col] = LabelEncoder().fit_transform(self.data[col].astype('str'))

        #print("Convert to numpy array")
        # def X and Y
        self.Y = self.data.iloc[:, [1]].values
        self.X = self.data.iloc[:, [2, 4, 5, 6, 7, 9]].values

        # Normalização Dataset
        # StandardScaler
        #print("Feature Scaling")
        self.X = StandardScaler().fit_transform(self.X)

        # StratifiedKFold
        #print("StratifiedKFold 5 splits")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        for train_index, test_index in skf.split(self.X, self.Y):
            self.X_train, self.X_test = self.X[train_index], self.X[test_index]
            self.Y_train, self.Y_test = self.Y[train_index], self.Y[test_index]

        return self.data, self.X_train, self.Y_train, self.X_test, self.Y_test
