# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    X_train = X_train.toarray()
    X = X.toarray()
    I1 = np.ones(shape=(X.shape[0], X.shape[1]))
    I2 = np.ones(shape=(X_train.shape[0], X_train.shape[1]))
    return (I1-X)@X_train.T + X@(I2-X_train).T


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    sorted_dist = np.argsort(Dist, axis=1, kind='margesort')
    return y[sorted_dist]


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    label_max = np.max(y) + 1
    result_matrix = []
    for i in range(y.shape[0]):
        line = np.bincount(y[i, range(k)], None, label_max)
        result_matrix.append([line[i] / k for i in range(0, label_max)])
    return np.array(result_matrix)


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    N = p_y_x.shape[0]
    M = p_y_x.shape[1]
    err_class = 0

    for i in range(0, N):
        if (M - np.argmax(p_y_x[i][::-1]) - 1) != y_true[i]:
            err_class += 1
    return err_class / N


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    dist = hamming_distance(X_val, X_train)
    y = sort_train_labels_knn(dist, y_train)

    errors = []
    for k in k_values:
        p_y_x = p_y_x_knn(y, k)
        err = classification_error(p_y_x, y_val)
        errors.append(err)

    k_pos = np.argmin(errors)
    best_k = k_values[int(k_pos % len(k_values))]
    best_error = errors[int(k_pos)]

    return best_error, best_k, errors


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    length = np.max(y_train)+1
    env = np.bincount(y_train, None, length)

    return env/len(y_train)


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    label_max = np.max(y_train) + 1
    denominator = np.bincount(y_train, None, label_max) + a + b - 2
    res = []

    def quotient(x):
        numerator = np.bincount(np.extract(x, y_train), None, label_max) + a - 1
        return numerator/denominator
    X_train=X_train.toarray()

    for i in range(X_train.shape[1]):
        res.append(quotient(X_train[:,i]))

    return np.array(res).T


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
    X = X.toarray()

    p_y_x = []
    for i in range(X.shape[0]):
        success = p_x_1_y ** X[i,]
        fail = (1 - p_x_1_y) ** (1-X)[i,]
        a = np.prod(success * fail, axis=1) * p_y
        # suma p(x|y') * p(y')
        sum = np.sum(a)
        # prawdopodobieñstwo każdej z klas podzielone przez sumę
        p_y_x.append(a / sum)
    return np.array(p_y_x)


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.

    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
    errors = []
    for a in a_values:
        error = []
        for b in b_values:
            p_y = estimate_a_priori_nb(y_train)
            p_x_1_y = estimate_p_x_y_nb(X_train, y_train, a, b)

            p_y_x = p_y_x_nb(p_y, p_x_1_y, X_val)
            err = classification_error(p_y_x, y_val)

            error.append(err)
        errors.append(error)

    err_pos = np.argmin(errors)
    index_a = int(err_pos/len(b_values))
    index_b = int(err_pos % len(a_values))

    error_best = errors[index_a][index_b]
    best_a = a_values[index_a]
    best_b = b_values[index_b]

    return error_best, best_a, best_b, errors