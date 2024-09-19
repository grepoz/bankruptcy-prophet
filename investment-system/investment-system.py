from data_utils import get_test_data

# we have a trained hybrid model
# we use test data to convey a simulation

hybrid_model = None

X_test, y_test, tickers_test = get_test_data()

for x, y, ticker in zip(X_test, y_test, tickers_test):
    prediction = hybrid_model.predict(x.values.reshape(1, -1))

    print(f'Prediction: {prediction[0]} Actual: {y}')

    if prediction[0] == y:
        print('Correct prediction')



    else:
        print(f'Incorrect prediction')
