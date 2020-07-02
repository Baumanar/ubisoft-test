# %%
import numpy as np
import time
import datetime
from scipy.spatial import cKDTree
from sklearn.metrics import precision_recall_fscore_support


def softmax(X, copy=True):
    """
    Calculate the softmax function.
    The softmax function is calculated by
    np.exp(X) / np.sum(np.exp(X), axis=1)
    This will cause overflow when large values are exponentiated.
    Hence the largest value in each row is subtracted from each data
    point to prevent this.
    Parameters
    ----------
    X : array-like of floats, shape (M, N)
        Argument to the logistic function
    copy : bool, optional
        Copy X or not.
    Returns
    -------
    out : array, shape (M, N)
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


# Helper function to
def datetime_to_timestamp(date_time):
    # Since all datapoint are created after the first january 2019, we can substract this timestamp for the timestamp to process
    return time.mktime(datetime.datetime.strptime(date_time, "%d/%m/%Y %H:%M").timetuple()) - time.mktime(
        datetime.datetime.strptime("01/01/2019 00:00", "%d/%m/%Y %H:%M").timetuple())


def get_df_features(data_frame):
    df_features = data_frame[
        # ["user_id", "order_created_datetime", \
         ["amount", "total_amount_14days", "email_handle_length", \
         "email_handle_dst_char", "total_nb_orders_player", "player_seniority", "geographic_distance_risk"]]
    # df_features['order_created_timestamp'] = df_features['order_created_datetime'].map(
    #     lambda x: datetime_to_timestamp(x))
    # df_features.pop("order_created_datetime")
    return df_features.to_numpy()


def get_df_labels(data_frame):
    return data_frame["transaction_status"].to_numpy()


# Helper function to give blocked labels pseudo labels based on the 1st nearest labeled neighbor
# Corresponds to step 3. in the ASSEMBLE.Adaboost algotrithm
def get_initial_blocked_labels(X, Y, X_blocked):
    kd_tree = cKDTree(X, compact_nodes=True)
    nearest_indexes = []
    for item in X_blocked:
        _, index_nearest = kd_tree.query(item, k=1)
        nearest_indexes.append(index_nearest)
    pseudo_labels = Y[nearest_indexes]
    return pseudo_labels


def print_metrics(y_test, predicted):
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
    given a confusion matrix (cm), make a nice plot
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
