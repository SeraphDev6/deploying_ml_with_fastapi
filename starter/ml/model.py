from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

models = {
    "RandomForestClassifier": RandomForestClassifier,
    "NeuralNetwork": MLPClassifier,
    "GaussianNB": GaussianNB,
    "KNeighborsClassifier": KNeighborsClassifier
}

def train_model(X_train, y_train, model_name, params = {}, gridsearch = False ):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    model_name : str
        Name of model that you want to train, should be one of the following:
            RandomForestClassifier
            NeuralNetwork
            GaussianNB
            KNeighborsClassifier
    params: dict
        A dictionary of keyword arguments to pass to the model, or a param grid if
        gridsearch is True (default={})
    gridsearch: bool
        A boolean indicating whether or not to use GridSearchCV (default=False)
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = models[model_name]
    if not gridsearch:
        rf = model(**params)
        return rf.fit(X_train,y_train)
    cv = GridSearchCV(model(), params)
    cv.fit(X_train,y_train)
    return cv.best_estimator_

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    
    return model.predict(X)
