import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from data_utils import get_train_data, get_test_data, get_stock_data
from hybrid_model import get_trained_models, predict_hybrid
from investment_agent import short_selling_strategy, get_share_count_to_borrow

import warnings
warnings.filterwarnings("ignore")

def make_simulation(random_state):

    tf_idf_vectorizer = TfidfVectorizer(max_features=1000)

    X_numerical_train, y_numerical_train, X_textual_train, y_textual_train = get_train_data()
    X_numerical_test, y_numerical_test, X_textual_test, y_textual_test, tickers_test, report_dates_test = get_test_data()

    decisionTreeClassifier, ada_boost_classifier = get_trained_models(
        X_numerical_train,
        y_numerical_train,
        X_textual_train,
        y_textual_train,
        tf_idf_vectorizer,
        random_state)

    X_textual_test_features = tf_idf_vectorizer.transform(X_textual_test['text'].values)

    # models are trained, now we can use them for prediction

    initial_money = 1000
    actual_money = initial_money

    empty_stock_list_errors_count = 0
    too_short_stock_data_errors_count = 0
    incorrect_predictions_count = 0
    tickers_with_error = {}

    j = 0

    print(f'Number of positive samples: {len(y_numerical_test[y_numerical_test == 1])}')

    for i in range(len(X_numerical_test)):

        j += 1

        y_numerical = y_numerical_test[i:i+1].values[0]

        # if not y_numerical:
        #     continue

        ticker, report_date = tickers_test[i:i+1], report_dates_test[i:i+1]

        hybrid_predictions = predict_hybrid(
            X_numerical_test[i:i + 1],
            X_textual_test_features[i],
            decisionTreeClassifier,
            ada_boost_classifier)

        pred = hybrid_predictions[0]

        if pred == False and y_numerical:
            incorrect_predictions_count += 1

        if y_numerical:
            pass

        if pred:

            if pred != y_numerical:
                incorrect_predictions_count += 1

            # report_date is the date of the report, but in real life it would be now()

            try:
                stock_data = get_stock_data(ticker, report_date)

            except Exception as e:
                print(f"Exception for {ticker}: {e}")
                tickers_with_error[str(ticker)] = report_date
                continue

            if stock_data.empty:
                # print(f"Stock data is empty for {ticker}")
                empty_stock_list_errors_count += 1
                tickers_with_error[str(ticker)] = report_date
                continue
            if len(stock_data) < 10:
                # print(f"Stock data is less than 10 days for {ticker}")
                too_short_stock_data_errors_count += 1
                tickers_with_error[str(ticker)] = report_date
                continue

            stock_data.ffill(inplace=True)

            historical_stock_prices = stock_data['close_price'][0:-10].values

            future_stock_prices_with_current_price = stock_data['close_price'][-11:].values

            current_price = historical_stock_prices[-1]

            aRIMAModel = auto_arima(historical_stock_prices,
                                    start_p=1, start_q=1,
                                    max_p=5, max_q=5,
                                    d=1,
                                    seasonal=False,
                                    stepwise=True,
                                    trace=False,
                                    suppress_warnings=True,
                                    maxiter=20,
                                    information_criterion='aic',
                                    random_state=random_state)

            arima_result = aRIMAModel.fit(historical_stock_prices)

            forecast = arima_result.predict(n_periods=10)

            short_selling_actions = short_selling_strategy(forecast, current_price)

            shares_count_to_borrow = get_share_count_to_borrow(current_price, actual_money)

            bought = False
            for day, (action, day_price) in enumerate(zip(short_selling_actions, future_stock_prices_with_current_price)):
                # print(f"Day {day}: {action}")

                if action == "BORROW-SELL" and not bought:
                    actual_money += (shares_count_to_borrow*day_price)
                    bought = True

                elif action == "BUY-RETURN" and bought:
                    actual_money -= (shares_count_to_borrow*day_price)
                    bought = False

            # print(f"Step: {j} | Final money: {actual_money}")


    # print(f"Incorrect predictions count: {incorrect_predictions_count}")

    return ((actual_money - initial_money)/initial_money)*100, incorrect_predictions_count

    # print(f"Empty stock list errors count: {empty_stock_list_errors_count}")
    # print(f"Too short stock data errors count: {too_short_stock_data_errors_count}")
    #
    #
    # # save tickers_with_error to file
    # with open('tickers_with_error.txt', 'w') as f:
    #     for ticker in tickers_with_error:
    #         f.write(f"{ticker}: {tickers_with_error[ticker]}\n")


profits = []
incorrect_predictions_counts = []
simulation_count = 5

for i in tqdm(range(simulation_count)):
    profit, incorrect_predictions_count = make_simulation(random_state=42+i)
    profits.append(profit)
    incorrect_predictions_counts.append(incorrect_predictions_count)

print(f"Profits: {profits}")
print(f"Average profit: {np.mean(profits)}")

print(f"Incorrect predictions counts: {incorrect_predictions_counts}")
print(f"Average incorrect predictions count: {np.mean(incorrect_predictions_counts)}")

# import matplotlib.pyplot as plt
# plt.plot(profits)
# plt.xlabel('Symulacja')
# plt.ylabel('Zysk %')
# plt.title(f'Procentowe zyski w czasie {simulation_count} symulacji')
# plt.show()
