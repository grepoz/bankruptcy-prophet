from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# def get_trained_models(X_numerical_train, y_numerical_train, X_textual_train, y_textual_train, tf_idf_vectorizer, random_state):
#     decisionTreeClassifier = DecisionTreeClassifier(
#         random_state=random_state,
#         min_samples_split=2,
#         min_samples_leaf=1,
#         max_depth=10,
#         criterion='entropy',
#         class_weight={True: 30, False: 1}
#     )
#     decisionTreeClassifier.fit(X_numerical_train, y_numerical_train)
#
#     X_textual_train_features = tf_idf_vectorizer.fit_transform(X_textual_train['text'].values)
#
#     ada_boost_classifier = AdaBoostClassifier(
#         random_state=random_state, n_estimators=100, learning_rate=1
#     )
#     ada_boost_classifier.fit(X_textual_train_features, y_textual_train)
#
#     return decisionTreeClassifier, ada_boost_classifier

def get_trained_models(X_numerical_train, y_numerical_train, X_textual_train, y_textual_train, tf_idf_vectorizer, random_state):
    decisionTreeClassifier = DecisionTreeClassifier(
        random_state=random_state,
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=10,
        criterion='entropy',
        class_weight={True: 30, False: 1}
    )
    decisionTreeClassifier.fit(X_numerical_train, y_numerical_train)

    X_textual_train_features = tf_idf_vectorizer.fit_transform(X_textual_train['text'].values)



    kNeighborsClassifier = KNeighborsClassifier(weights='uniform', p=2, n_neighbors=7, leaf_size=10, algorithm='brute')
    kNeighborsClassifier.fit(X_textual_train_features, y_textual_train)

    return decisionTreeClassifier, kNeighborsClassifier


# def predict_hybrid_old(X_numerical, X_textual, decisionTreeClassifier, adaBoostClassifier):
#
    # X_numerical = X_numerical.values
    #
    # if X_numerical.ndim == 1:
    #     X_numerical = X_numerical.reshape(1, -1)
    #
    # X_textual = X_textual.reshape(1, -1)
#
#     numerical_preds = decisionTreeClassifier.predict_proba(X_numerical)[:, 1]
#     textual_preds = adaBoostClassifier.predict_proba(X_textual)[:, 1]
#
#     decisionTreeClassifier_weight, adaBoostClassifier_weight = 0.2, 0.8
#
#     final_preds = decisionTreeClassifier_weight * numerical_preds + adaBoostClassifier_weight * textual_preds
#     final_preds = final_preds > 0.5
#
#     return final_preds


def predict_hybrid(X_numerical, X_textual, decisionTreeClassifier, kNeighborsClassifier,
                   decisionTreeClassifier_weight=0.4, kNeighborsClassifier_weight=0.6):

    X_numerical = X_numerical.values

    if X_numerical.ndim == 1:
        X_numerical = X_numerical.reshape(1, -1)

    X_textual = X_textual.reshape(1, -1)

    numerical_preds = decisionTreeClassifier.predict_proba(X_numerical)[:, 1]
    textual_preds = kNeighborsClassifier.predict_proba(X_textual)[:, 1]

    final_preds = decisionTreeClassifier_weight * numerical_preds + kNeighborsClassifier_weight * textual_preds
    final_preds = final_preds > 0.5

    return final_preds
