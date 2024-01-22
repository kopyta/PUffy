import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression




class LabellingMethods():
    """
    Class containing methods for relabelling positive instances as unlabelled.

    The new labelling is stored as a new feature 's', where positives have a value '1' 
    and initially assumed unlabelled instances have a value '0'.
    """
    
    def __init__(self, probability: int, a_net: np.array = np.arange(-20, 20, 0.05)) -> None:

        '''
        Parameters
        ----------
        probability :  np.array
            Probability with which positive samples will be selected for the new feature 's'.
        a_net : np.array
            Grid to search the best value a in SAR methods to come out on the right probability .
        '''
        self.probability = probability
        self.a_net = a_net


    def scar(self, y: pd.Series) -> np.array:
        """
        Applies the SCAR method to relabel positive instances as unlabeled based on positive label frequency.

        The SCAR method chooses positive instances to be labeled as unlabeled based solely on the positive label frequency in the data.
        The sample is defined by the propensity score 'c', which is constant and equal to the positive label frequency.

        Parameters
        ----------
        y : pd.Series
            Target variable containing the original labels.
        probability : float
            Probability with which positive samples will be selected for the new labeling.

        Returns
        -------
        np.array
            Array containing the new labels for positive instances.
        """
        s = [1 if i == 1 and np.random.uniform() < self.probability else 0 for i in y]
        return np.array(s)
    

    def sar_sigmoid(self, X: pd.DataFrame, y: pd.Series) -> np.array:
        """
        This method chooses positive instances to be labeled as unlabeled based on the attributes.
        Probability is calculated with sigmoid function.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target variable.

        Returns
        -------
        Numpy array
            np.array containing the new labels for positive instances.
        """

        fractions = []
        if len(X.index) != len(y):
            raise ValueError("Vectors are different lenght.")
        b = np.ones(len(X.iloc[0]))
        for a in self.a_net:
            s = np.zeros(len(y))
            for i in range(len(y)):
                if y[i] != 0:
                    val = a + np.dot(b, X.iloc[i])
                    prob = 1 / (1 + np.exp(-val))
                    if np.random.uniform() < prob:
                        s[i] = 1
            f = sum(s) / sum(y)
            fractions.append([a, f, s])
        s = min(fractions, key=lambda x: abs(x[1] - self.probability))[2]
        print(f"fraction is {min(fractions, key=lambda x: abs(x[1] - self.probability))[1]}")
        print(f"a = {min(fractions, key=lambda x: abs(x[1] - self.probability))[0]}")
        return np.array(s)
    
    def sar_cauchy(self, X: pd.DataFrame, y: pd.Series) ->  np.array:
        """
        This method chooses positive instances to be labeled as unlabeled based on the attributes.
        Probability is calculated with Cauchy distribution.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target variable.

        Returns
        -------
        Numpy array
            np.array containing the new labels for positive instances.
        """

        fractions = []
        if len(X.index) != len(y):
            raise ValueError("Vectors are different lenght.")
        b = np.ones(len(X.iloc[0]))
        for a in self.a_net:
            s = np.zeros(len(y))
            for i in range(len(y)):
                if y[i] != 0:
                    val = a + np.dot(b, X.iloc[i])
                    prob = 1/np.pi * np.arctan((val - 0) / 1) + 0.5
                    if np.random.uniform() < prob:
                        s[i] = 1
            f = sum(s) / sum(y)
            fractions.append([a, f, s])
        s = min(fractions, key=lambda x: abs(x[1] - self.probability))[2]
        print(f"fraction is {min(fractions, key=lambda x: abs(x[1] - self.probability))[1]}")
        print(f"a = {min(fractions, key=lambda x: abs(x[1] - self.probability))[0]}")
        return np.array(s)

    def sar_lr_sigmoid(self, X: pd.DataFrame, y: pd.Series) ->  np.array:
        """
        This method chooses positive instances to be labeled as unlabeled based on the attributes.
        Observations more closely positive class are assigned the label 1 with higher probability.
        Probability is calculated withsigmoid function.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target variable.

        Returns
        -------
        Numpy array
            np.array containing the new labels for positive instances.
        """

        model = LogisticRegression().fit(X, y)
        coefficients = model.coef_
        fractions = []
        if len(X.index) != len(y):
            raise ValueError("Vectors are different lenght.")
        for a in self.a_net:
            s = np.zeros(len(y))
            for i in range(len(y)):
                if y[i] != 0:
                    val = a + np.dot(coefficients, X.iloc[i])
                    prob = 1 / (1 + np.exp(-val))
                    if np.random.uniform() < prob:
                        s[i] = 1
            f = sum(s) / sum(y)
            fractions.append([a, f, s])
        s = min(fractions, key=lambda x: abs(x[1] - self.probability))[2]
        print(f"fraction is {min(fractions, key=lambda x: abs(x[1] - self.probability))[1]}")
        print(f"a = {min(fractions, key=lambda x: abs(x[1] - self.probability))[0]}")
        return np.array(s)

    