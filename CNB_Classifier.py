import random
from unittest import result
import pandas as pd
import numpy as np
from math import sqrt, exp, pi
from lifelines import KaplanMeierFitter
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored
from lifelines.datasets import load_rossi

class CNB:
    unique_sorted_times = None
    n_cols = None
    n_rows = None
    params = {}
    St = KaplanMeierFitter()
    G = KaplanMeierFitter()

    def __init__(self):
        pass

    def fit(self, X, T, δ):
        T = np.array(T)
        δ = np.array(δ)
        self.n_rows = len(X)
        X = pd.DataFrame(X)
        self.n_cols = len(X.columns)
        self.unique_sorted_times = pd.Series(T).unique()
        self.unique_sorted_times.sort()
        self.St.fit(T, δ)
        self.G.fit(T, 1 - δ)

        # calculate params
        for t in self.unique_sorted_times:
            self.params[t] = {}

            # calculate Y_i
            Y_i = (T > t).astype(int)

            sum_y_i = sum(Y_i)

            # calculate w
            w_i = np.array([0 if δ[i] == 0 and t > T[i] else 1 /
                            max(1e-5,self.G.predict(min(time, t))) for i, time in enumerate(T)])

            w_δ_y = w_i*δ*(1-Y_i)
            sum_w_δ_y = sum(w_δ_y)

            for j in range(0, self.n_cols):
                self.params[t][j] = np.array([0, 1, 0, 1])

                # calculate μ
                if sum_y_i != 0:
                    self.params[t][j][0] = sum(Y_i*X.iloc[:,j])/sum_y_i

                # calculate σ
                if sum_y_i != 0:
                    self.params[t][j][1] = sqrt(
                        (sum(Y_i*(X.iloc[:,j]**2))/sum_y_i) -
                        (self.params[t][j][0]**2)
                    )
                    self.params[t][j][1] = self.params[t][j][1] if self.params[t][j][1] != 0 else 1
    

                # calculate θ
                if sum_w_δ_y != 0:
                    self.params[t][j][2] = sum(w_δ_y*X.iloc[:,j])/sum_w_δ_y

                # calculate ψ
                if sum_w_δ_y != 0:
                    self.params[t][j][3] = sqrt(
                        (sum(w_δ_y*(X.iloc[:,j]**2))/sum_w_δ_y) -
                        (self.params[t][j][2]**2)
                    )
                    self.params[t][j][3] = self.params[t][j][3] if self.params[t][j][3] != 0 else 1
                    

    def S(self, T):
        return np.array(self.St.predict(T))

    def calculate_probability(self, x, mean, stdev):
        exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    def psi(self, X, T):
        X = pd.DataFrame(X)
        T = np.array(T)
        prediction = list()
        for i, ts in enumerate(T):
            # clamp to the closest computed t value
            diffs = abs(self.unique_sorted_times - ts)
            index = np.where(diffs == diffs.min())[0][0]
            t = self.unique_sorted_times[index]
            f_gt = np.prod([self.calculate_probability(X.iloc[i, j], self.params[t][j][0], self.params[t][j][1])
                         for j in range(0, self.n_cols)])
            prediction.append(
                 f_gt / (f_gt*self.S(t) +
                    np.prod([self.calculate_probability(X.iloc[i, j], self.params[t][j][2], self.params[t][j][3])
                             for j in range(0, self.n_cols)])*(1-self.S(t))
                )
            )
        return prediction

    def predict(self, X, T):
        return self.S(T)*self.psi(X, T)