# %%
import numpy as np
import time
import datetime
from scipy.spatial import cKDTree
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter


def transform_user_id(X, user_id_counter):
    """
    Transforms in place an array of user ids
    :param X: array of user ids
    :param user_id_counter: user_id_counter dictionnary
    """
    for i, elem in enumerate(X):
        if elem in user_id_counter:
            X[i] = user_id_counter[elem]
        else:
            X[i] = 1


def transform_order_date(X):
    """
    Transforms in place an array of dates
    :param X: array of dates
    """
    for i, elem in enumerate(X):
        X[i] = int(elem.split('/')[0])


def preprocess(X, user_id_counter):
    """
    Transforms both user ids and dates in a 2D array
    :param X: Array of features
    :param user_id_counter: user_id_counter dictionnary
    """
    transform_user_id(X[:, 0], user_id_counter)
    transform_order_date(X[:, 1])


def get_df_features(data_frame):
    """
    Transforms dataframe with wanted features to a numpy array
    :param data_frame: input dataframe
    :return: numpy array of features
    """
    df_features = data_frame[
        ["user_id", "order_created_datetime", "amount", "total_amount_14days", "email_handle_length", \
         "email_handle_dst_char", "total_nb_orders_player", "player_seniority", \
         "total_nb_play_sessions", "geographic_distance_risk"]]
    return df_features.to_numpy()


def get_df_labels(data_frame):
    """
    Get labels of examples in the dataframe
    :param data_frame: input dataframe
    :return: numpy array of labels [-1, 1]
    """
    y_labeled = data_frame["transaction_status"].to_numpy()
    return np.array([1 if elem == "LEGIT" else -1 for elem in y_labeled])


def get_initial_blocked_labels(X, Y, X_blocked):
    """
    Helper function to give blocked labels pseudo labels based on the 1st nearest labeled neighbor.
    Corresponds to step 3. in the ASSEMBLE.Adaboost algotrithm
    Relies on sklearn's kdtree to compute quickly nearest neighbors
    :param X: Labeled features
    :param Y: Labels of the features
    :param X_blocked: Blocked features
    :return: pseudo labels of blocked features
    """
    kd_tree = cKDTree(X, compact_nodes=True)
    nearest_indexes = []
    for item in X_blocked:
        _, index_nearest = kd_tree.query(item, k=1)
        nearest_indexes.append(index_nearest)
    pseudo_labels = Y[nearest_indexes]
    return pseudo_labels


def softmax(X, copy=True):
    """
    Calculate the softmax function.
    ----------
    :param X: array-like of floats, shape (M, N)
    :param copy: bool, optional
        Copy X or not.
    Returns
    -------
    :return: array, shape (M, N)
        Softmax function evaluated at every point in x
    """
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X


def optimal_decision(amount, fraud_fee, p):
    """
    Computes the optimal decision between blocking or not blocking of the transaction
        Parameters
    :param amount: amount of the transaction
    :param fraud_fee: fraud fee
    :param p: probability of the transaction to be a fraud
    :return: string :
            BLOCK if the transaction must be blocked
            ACCEPT if the can be accepted
    """
    potential_profit = amount * (1 - p) - p * fraud_fee
    if potential_profit < 0:
        return 'BLOCK'
    else:
        return 'ACCEPT'

def print_metrics(y_test, predicted):
    """
    Prints precision, recall and f1-score of predicted results
    :param y_test: true labels
    :param predicted: predcited labels
    :return:
    """
    precision, recall, f_score, _ = precision_recall_fscore_support(y_test, predicted)
    print('precision: FRAUD: {:.4f}  LEGIT: {:.4f}'.format(precision[0], precision[1]))
    print('recall:    FRAUD: {:.4f}  LEGIT: {:.4f}'.format(recall[0], recall[1]))
    print('f1-score:    FRAUD: {:.4f}  LEGIT: {:.4f}'.format(f_score[0], f_score[1]))


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    Plot confusion matrix
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
