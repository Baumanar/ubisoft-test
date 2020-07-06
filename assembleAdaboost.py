from sklearn.tree import DecisionTreeClassifier
import numpy as np
from utils import softmax


class AssembleAdaBoost:

    def __init__(self, n_estimators=50, beta=0.9, alpha=1.0):
        """
        :param n_estimators: number of weak learners
        :param beta: sample weight parameter for unlabeled data
        :param alpha: step size
        """
        self.n_estimators = n_estimators
        self.beta = beta
        self.alpha = alpha

        self.stumps = np.zeros(shape=n_estimators, dtype=object)
        self.stump_weights = np.zeros(shape=n_estimators)

    def fit(self, X_labeled, X_blocked, y_labeled, y_blocked):
        """
        fit methods trains the AssembleAdaboost on the provided data. Blocked inputs must have been
        labeled.
        :param X_labeled: Labeled input features
        :param X_blocked: Unlabeled inputs features (blocked)
        :param y_labeled: Labels of the inputs
        :param y_blocked: Labels of the blocked inputs
        """
        num_labeled = X_labeled.shape[0]
        num_blocked = X_blocked.shape[0]
        # Initialize sample weights for labeled and unlabeled data (step 2 in the algorithm)
        sample_weights = np.array([self.beta / num_labeled for i in range(num_labeled)] \
                                  + [(1 - self.beta) / num_blocked for i in range(num_blocked)])
        # Concatenate labeled and un labeled inputs to create the entire training dataset and labels
        X = np.concatenate([X_labeled, X_blocked])
        y = np.concatenate([y_labeled, y_blocked])

        for t in range(self.n_estimators):
            # Fit  weak learner with current sample weights (step 5)
            stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
            stump = stump.fit(X, y, sample_weight=sample_weights)

            # Compute predictions of the weak learner (step 6)
            stump_pred = stump.predict(X)
            # Compute epsilon error (step 7)
            err = sample_weights[(stump_pred != y)].sum()

            if err > 0.5:  # (step 8)
                self.stumps = self.stumps[self.stumps != 0]
                self.stump_weights = self.stump_weights[self.stump_weights != 0]
                return

            # Compute weight of the weak learner (step 9)
            stump_weight = 0.5 * np.log((1 - err) / err)

            # Add the current stump and its weight to the ensemble of weak learners
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            # Compute predictions (F_t) of the classifier ensemble so far (step 10)
            pred_t = np.array([stump.predict(X) for stump in self.stumps[0:t + 1]])
            pred_t = np.dot(self.stump_weights[0:t + 1], pred_t)

            # Get new pseudo-labels for blocked inputs (step 11 in the algorithm)
            y_blocked = np.sign(pred_t[-num_blocked:])
            y = np.concatenate([y_labeled, y_blocked])

            # Update sample weights
            sample_weights = self.alpha * np.exp(-pred_t * y)
            sample_weights /= sample_weights.sum()

    def predict_proba(self, X):
        """
        Predict class probabilities for X
        :param X: array-like, sparse matrix} of shape (n_samples, n_features)
        :return: ndarray of shape (n_samples, n_classes)
        """
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        decision = np.dot(self.stump_weights, stump_preds)
        decision = np.vstack([-decision, decision]).T / 2
        return softmax(decision, copy=False)

    def predict(self, X):
        """
        Predict classes for X
        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
        :return: ndarray of shape (n_samples,)
        """
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        return np.sign(np.dot(self.stump_weights, stump_preds))
