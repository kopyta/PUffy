import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial import distance
from scipy.stats import norm

from typing import Tuple, Union,List
import itertools, numbers


class RnMethods():
    """
    Class for identifying reliable negatives using various methods.
    """
    
    def convert_to_list(self,a: Union[list, pd.Series, int, float, str, tuple, np.ndarray]) -> List:
        if isinstance(a, (list, pd.Series)):
            return a
        elif isinstance(a, numbers.Number):
            # Convert single values to a list
            return [int(a)]

        elif isinstance(a, str):
            try:
                a = [int(a)]  # Try to convert string to int
            except ValueError:
                print(f"Warning: Unable to convert {a} to int. Returning as a list.")
                return [a]
            return a

        elif isinstance(a, (tuple, np.ndarray)):
            # Convert tuples and numpy arrays to a list
            return list(a)
        else:
            # For other types, print a warning and return a list
            print(f"Warning: {a} must be an instance of list, pd.Series, int, float, str, tuple, or np.ndarray.")
            return [a]

    def compare_cluster_dis(self,df: pd.DataFrame,groupA: np.array, groupB: np.array) -> int:
        """
        Finds the furthest cluster from groupA from clusters in groupB based on davies_bouldin_score.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with instances assigned to clusters.

        groupA : np.array
            List of clusters to choose from.

        groupB : np.array
            List of clusters to compare to.

        Returns
        -------

        int
            Number of the furthest cluster.
        """
    
        groupA,groupB = [self.convert_to_list(i) for i in [groupA,groupB]]

        groupB_df = df[df['cluster'].isin(groupB)]
        db_scores = []
        for i in groupA:
            groupA_df =df[df['cluster']==i]
            compare_distance_df = pd.concat([groupB_df, groupA_df]).drop('cluster', axis=1)
            labels_kmeans = df.loc[compare_distance_df.index]['cluster']
            db_scores.append(davies_bouldin_score(compare_distance_df,labels_kmeans))
        return int(groupA[np.argmin(db_scores)])

    def kmeans(self, X: pd.DataFrame, s: np.array, random_state: bool=None, verbose=0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform K-means clustering on the input features with optimized hyperparameters.

        Parameters
        ----------
        X : pd.DataFrame
            Input matrix of features.

        s : np.array
            Indicator variable of labeled instances.

        random_state : bool, default=None
            Random state to initiate KMeans clusterization.

        verbose : bool, default='False'
            Indicator whether to print more information.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A tuple of three DataFrames representing:
            - reliable negatives (rn)
            - labeled instances (p)
            - unlabeled instances (u)

        Example
        -------
        >>> from puffy.labelling_methods import LabellingMethods
        >>> from puffy.rn_methods import RnMethods
        >>> from sklearn.datasets import load_breast_cancer
        >>>       
        >>> X, y = load_breast_cancer(as_frame=True, return_X_y=True)
        >>>
        >>> Labeler = LabellingMethods(probability=0.9)
        >>> labels = Labeler.sar_lr_sigmoid(X,y)
        >>>
        >>> RnIdentifier = RnMethods()
        >>> rn,l,u = RnIdentifier.kmeans(X,labels)
        Fraction is 0.8991596638655462
        a = -1.250000000000024
        KMEANS
        Best score: 0.6752581310732739. Best parameters: {'n_clusters': 3, 'n_init': 15, 'algorithm': 'auto'}
        Cluster share in the whole dataset: 
        ones     0.000000
        zeros    3.339192
        Name: 2, dtype: float64
        KMEANS: There were found 19 reliable negatives from 248 unlabeled instances
        """
        print("KMEANS")
        param_grid = {
            'n_clusters': list(range(2, 7)),
            'n_init': ['auto', 5, 10, 15],  
            'algorithm': ['auto', 'full', 'elkan']  
            }

        best_clusterer = None
        best_score = -1  # min silhouette score value
        best_parameters = None
        if 's' in X.columns:
            X.drop('s', axis=1, inplace=True)
        run = 1
        for parameters_set in list(itertools.product(*param_grid.values())):
            params = dict(zip(param_grid.keys(), parameters_set))
            # clustering
            clusterer = KMeans(**params,random_state=random_state)
            cluster_labels = clusterer.fit_predict(X)
            # evaluation
            silhouette_mean = silhouette_score(X, cluster_labels)
            if silhouette_mean >= best_score:
                best_score = silhouette_mean
                best_clusterer = clusterer
                best_parameters = params
            run += 1

        print(f'Best score: {best_score}. Best parameters: {str(best_parameters)}')

        df = X.copy()
        df['s'] = s
        df['cluster'] = best_clusterer.predict(X)

        cluster_counts = df.groupby('cluster')['s'].value_counts().unstack(fill_value=0)

        # Find the claster with the most ones, and the cluster the furtherst from it
        max_ones_cluster = cluster_counts[cluster_counts[1] == cluster_counts[1].max()].index.values
        print(f"Initial choice of the most positive cluster{max_ones_cluster}") if verbose ==1 else None
        not_max_ones_cluster = [cluster for cluster in range(best_clusterer.n_clusters) if cluster not in list(max_ones_cluster)]
        print(f"Initial chocice of not the most positive clusters: {not_max_ones_cluster}") if verbose ==1 else None

        if len(max_ones_cluster)>1:
            max_ones_cluster = self.compare_cluster_dis(df,max_ones_cluster,not_max_ones_cluster)
            min_ones_cluster = self.compare_cluster_dis(df, not_max_ones_cluster, max_ones_cluster)
            print(f"Found more than one most positive cluster, choosing the one the furthest from the rest:{max_ones_cluster}\n\
                Supposed reliable negatives cluster: {min_ones_cluster}") if verbose ==1 else None

        else:
            min_ones_cluster = self.compare_cluster_dis(df, not_max_ones_cluster, max_ones_cluster)
            print(f"There's only one cluster with the most positives, found the cluster furthest from it: {min_ones_cluster}") if verbose ==1 else None


        if isinstance(min_ones_cluster, list):
            min_ones_cluster = cluster_counts[cluster_counts.loc[min_ones_cluster][0]>0][0]
            print(f"Found more than one cluster the furthest from the positive cluster, chosing the one with unlabaled instances or else the first one: {min_ones_cluster}") if verbose ==1 else None

        groupB = [max_ones_cluster[0],min_ones_cluster]
        print(f"the positive cluster, the furthest cluster from it: {groupB}") if verbose ==1 else None

        while cluster_counts.loc[min_ones_cluster][0] == 0 and len(groupB)!=best_clusterer.n_clusters:
            print(f"The reliable negative cluster lacks unlabeled instances, choosing another") if verbose ==1 else None
            groupA = [cluster for cluster in range(best_clusterer.n_clusters) if cluster not in groupB]
            min_ones_cluster = self.compare_cluster_dis(df, groupA, groupB)
            groupB.append(min_ones_cluster)
            
        if cluster_counts.loc[min_ones_cluster][0]==0:
            min_ones_cluster = max_ones_cluster[0]
            raise Warning("The found cluster contains all of the unlabeled instances, all unlabeled are considered reliable negatives")

        print(f"The chosen reliable negatives cluster is {min_ones_cluster}") if verbose ==1 else None
        print("Cluster share in the whole dataset: ") if verbose ==1 else None
        print(100* cluster_counts.loc[min_ones_cluster]/ len(X)) if verbose ==1 else None

        rn = df[df['cluster'] == min_ones_cluster].copy()
        rn = rn[rn['s'] != 1]
        p = df[df['s'] == 1]
        u_all = df[df['s'] == 0]
        u = u_all.loc[~u_all.index.isin(rn.index)]


        print("KMEANS: There were found", len(rn), "reliable negatives from", len(u_all), "unlabeled instances")
        return rn.iloc[:,:-1], p.iloc[:,:-1], u.iloc[:,:-1]
    
    def calc_logl(self, x: float, mu: float, sd: float) -> float:
        """
        Vectorized function to calculate log-likelihood.

        Parameters
        ----------
        x : float
            Input value.

        mu : float
            Mean.

        sd : float
            Standard deviation.

        Returns
        -------
        float
            Log-likelihood value.
        """
        logl = np.sum(np.log(norm.pdf(x, mu, sd)))
        return logl

    def find_optimal_idx(self, data: np.array) -> int: 
        """
        Find the optimal index to use as a cut-off in a numpy array based on log-likelihood.

        Parameters
        ----------
        data : np.array
            Input array.

        Returns
        -------
        int
            Optimal index.
        """
                
        profile_logl = []
        for q in range(1,len(data)):
            n = len(data)
            s1 = data[0:q]
            s2 = data[q:]
            mu1 = s1.mean()
            mu2 = s2.mean()
            sd1 = s1.std()
            sd2 = s2.std()
            sd_pooled = np.sqrt((((q-1)*(sd1**2)+(n-q-1)*(sd2**2)) / (n-2)))
            profile_logl.append(self.calc_logl(s1,mu1,sd_pooled) + self.calc_logl(s2,mu2,sd_pooled))

        return np.argmax(profile_logl)
        
    def knn(self, X: pd.DataFrame, y: np.array, s: np.array, k: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Use k-Nearest Neighbors to identify reliable negatives, positives, and unlabeled examples.

        Parameters
        ----------
        X : pd.DataFrame
            Input matrix of features.

        y : np.array
            Indicator variable for examples to be positive (1).

        s : np.array
            Indicator variable for examples to be labeled (1 for positive, 0 for unlabeled).

        k : int, optional
            Number of neighbors to consider (default is 5).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A tuple of three DataFrames representing:
            - Reliable negatives (rn)
            - Positives (p)
            - Unlabeled (u)
        """
        print("KNN")
        X_copy = X
        X_copy['s'] = s

        # split dataset into positives and unlabeled

        P = (X_copy.iloc[np.where(s == 1)[0]]).to_numpy()
        U = (X_copy.iloc[np.where(s == 0)[0]]).to_numpy()

        # calculate the distance to k nearest positives for each unlabled observation
        distances = distance.cdist(U, P)
        k_nearest_indices = np.argpartition(distances, k, axis=1)[:, :k]
        dist = np.sum(np.take_along_axis(distances, k_nearest_indices, axis=1), axis=1)

        # sort distances
        sorted_dist = sorted(dist)

        # find optimal idx
        idx = self.find_optimal_idx(np.array(sorted_dist))
        indices = [i for i, value in enumerate(dist) if value > sorted_dist[idx]]

        # get datasets
        rn = X_copy.iloc[np.where(s == 0)[0]].iloc[indices]
        p = X_copy.iloc[np.where(s == 1)[0]]
        all_u = X_copy.iloc[np.where(s == 0)[0]]
        u = all_u.loc[~all_u.index.isin(rn.index)]

        rn_s = np.array(rn['s'])
        rn_y = y[rn.index]

        tn = np.sum((rn_y == 0) & (rn_s == 0))
        fn = np.sum((rn_y == 1) & (rn_s == 0))

        print("KNN: There were found", len(rn), "reliable negatives from", len(U), "unlabeled")

        return rn, p, u

    def choose_spies(self, s: np.array, percent: int) -> Tuple[np.array, np.array]:
        """
        Draw a given percentage of positive samples to act as spies. Returns new labels and spy indices.

        Parameters
        ----------
        s : np.array
            Indicator variable for examples to be labeled (1 for positive, 0 for unlabeled).

        percent : int
            Percentage of positive samples to act as spies.

        Returns
        -------
        Tuple[np.array, np.array]
            A tuple containing:
            - New labels with spies (spies)
            - Indices of spy samples (cast_lots)
        """
        num_of_spy = int(sum(s) * percent / 100)

        # cast_lots are spies index
        unlabeled = np.where(s == 0)[0]
        cast_lots = np.random.choice(np.where(s == 1)[0], num_of_spy, replace=False)
        spies = s.copy()
        spies[cast_lots] = 0

        return spies, cast_lots
    
    def spy(self, X: pd.DataFrame, s: np.array, percent: int = 15, show_treshhold = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Identify reliable negatives using a spy model. Returns reliable negatives, positives, and unlabeled examples.

        Parameters
        ----------
        X : pd.DataFrame
            Input matrix of features.

        s : np.array
            Indicator variable for examples to be labeled (1 for positive, 0 for unlabeled).

        percent : int, optional
            Percentage of positive samples to act as spies (default is 15).

        Returns:
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A tuple of three DataFrames representing:
            - Reliable negatives (rn)
            - Positives (p)
            - Unlabeled (u)
        """
        print("SPY")
        X_df = X.copy()
        
        spies, cast_lots = self.choose_spies(s, percent)
        X_df['s'] = s
        positive = X_df.iloc[np.where(s == 1)[0]].copy()
        model = LogisticRegression(max_iter=10000)
        model.fit(X_df.drop(['s'], axis = 1), spies)
        predictions = model.predict_proba(X_df.drop(['s'], axis = 1))
        prob_class_1 = predictions[:, 1]
        threshold = min(prob_class_1[cast_lots])
        if show_treshhold:
            print(f"Treshhold is {threshold}")
        y_predict_class = [1 if prob >= threshold else 0 for prob in prob_class_1]
        X_df['y_predict_class'] = y_predict_class

        unlabeled = X_df.loc[(X_df['s'] == 0) & (X_df['y_predict_class'] == 1)].drop(['y_predict_class'], axis = 1)
        rn = X_df.loc[(X_df['s'] == 0) & (X_df['y_predict_class'] == 0)].drop(['y_predict_class'], axis = 1)

        print("SPY: There were found", len(rn), "reliable negatives from", len(s)-sum(s), "unlabeled")
        return rn, positive, unlabeled
    
