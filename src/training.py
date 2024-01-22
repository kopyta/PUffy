import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class Train():
    """
    Class for training logistic regression models using different methods.
    """

    def iterative_LR(self, X_df: pd.DataFrame, rn: pd.DataFrame, p: pd.DataFrame, u: pd.DataFrame, s_label: np.array, show_results_flag = True, epoch: int = 12) -> LogisticRegression:
        """
        Trains a logistic regression model iteratively with a specified number of iterations.

        Parameters
        ----------
        X_df : pd.DataFrame
            Features dataframe.
        rn : pd.DataFrame
            Reliable negatives dataframe.
        p : pd.DataFrame
            Positives dataframe.
        u : pd.DataFrame
            Unlabeled dataframe.
        s_label : np.array
            Artificial PU labels
        epoch : int, optional
            Number of iterations (default is 12).

        Returns
        -------
        LogisticRegression
            Trained logistic regression model.
        """
        X_df['s'] = s_label
        global unlabeled_count
        unlabeled_count = len(X_df) - (len(p) + len(rn))
        max_iterations = epoch

        print("Start iterative training of logistic regression model.")

        def stop(iteration, unlabeled):
            global unlabeled_count
            if iteration == max_iterations:
                return True
            if unlabeled == 0:
                return True
            if unlabeled == unlabeled_count:
                return True
            unlabeled_count = unlabeled

        iteration = 1

        iter_results = []

        while True:
            
            model = LogisticRegression(random_state=0)
            iteration_X = pd.concat((rn.iloc[:, :-1], p.iloc[:, :-1]), axis=0)
            iteration_y = pd.concat((rn['s'], p['s']), axis=0)

            # Trening
            model.fit(iteration_X, iteration_y)

            if u.shape[0] == 0:
                return model

            # Predykcja
            labeled_instances_indices = np.concatenate((rn.index, p.index))
            u = X_df.loc[~X_df.index.isin(labeled_instances_indices)]
            pred = model.predict(u.iloc[:, :-1])

            # Rozszerzenie zbioru reliable negatives
            new_rn = u.loc[~pred.astype(bool)]
            rn = pd.concat((rn, new_rn), axis = 0)

            # Zawężenie zbioru unlabeled
            new_u = u.loc[pred.astype(bool)]
            u = new_u

            unlabeled = len(u)
            iter_results.append([len(iteration_y),len(rn), len(p), len(u)])
            assert len(rn) + len(p) + len(u) == len(X_df)

            if stop(iteration, unlabeled):
                print("Stopping")
                print("")
                break
            iteration += 1

        iterations_df = pd.DataFrame(iter_results, columns =['Training', 'RN', 
                                                             'P', 'U'])
        if show_results_flag:
            print(iterations_df)
        return model
    
    def naive_method(self, X: pd.DataFrame, s: np.array) -> LogisticRegression:
        """
        Trains a logistic regression model using the naive method.

        Parameters
        ----------
        X : pd.DataFrame
            Features dataframe.
        s : np.array
            Labels for the 's' feature.

        Returns
        -------
        LogisticRegression
            Trained logistic regression model.

        """
        model = LogisticRegression(max_iter=10000)
        model.fit(X, s)

        return model

