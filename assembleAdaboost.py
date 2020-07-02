from sklearn.tree import DecisionTreeClassifier
import numpy as np
from utils import softmax
from collections import Counter


class AssembleAdaBoost:

    def __init__(self, n_estimators=50, beta=0.9, alpha=1.0):
        self.n_estimators = n_estimators
        self.beta = beta
        self.alpha = alpha

        self.stumps = np.zeros(shape=n_estimators, dtype=object)
        self.stump_weights = np.zeros(shape=n_estimators)

    def fit(self, X_labeled, X_blocked, y_labeled, y_blocked):
        num_labeled = X_labeled.shape[0]
        num_blocked = X_blocked.shape[0]

        sample_weights = np.array([self.beta / num_labeled for i in range(num_labeled)] \
                                  + [(1 - self.beta) / num_blocked for i in range(num_blocked)])
        X = np.concatenate([X_labeled, X_blocked])
        y = np.concatenate([y_labeled, y_blocked])

        for t in range(self.n_estimators):

            # fit  weak learner
            stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump = stump.fit(X, y, sample_weight=sample_weights)

            # calculate error and stump weight from weak learner prediction
            stump_pred = stump.predict(X)
            # calculate epsilon error
            err = sample_weights[(stump_pred != y)].sum()

            if err > 0.5:  # (step 8)
                self.stumps = self.stumps[self.stumps != 0]
                self.stump_weights = self.stump_weights[self.stump_weights != 0]
                return

            stump_weight = 0.5 * np.log((1 - err) / err)

            # add the current stump and its weight to the ensemble of weak learners
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            # Compute predictions (F_t) of the classifier ensemble so far (step 10 in the algorithm)
            pred_t = np.array([stump.predict(X) for stump in self.stumps[0:t + 1]])
            pred_t = np.dot(self.stump_weights[0:t + 1], pred_t)
            # Get new pseudo-labels for blocked inputs (step 11 in the algorithm)

            y_blocked = np.sign(pred_t[-num_blocked:])
            y = np.concatenate([y_labeled, y_blocked])

            # update sample weights
            sample_weights = self.alpha * np.exp(-pred_t * y)
            sample_weights /= sample_weights.sum()

    def predict_proba(self, X):
        """ Make predictions using already fitted model """
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        decision = np.dot(self.stump_weights, stump_preds)
        decision = np.vstack([-decision, decision]).T / 2

        return softmax(decision, copy=False)

    def predict(self, X):
        """ Make predictions using already fitted model """
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        return np.sign(np.dot(self.stump_weights, stump_preds))
