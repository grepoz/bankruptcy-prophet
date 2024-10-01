def short_selling_strategy(forecasted_prices, current_price):

    def calculate_short_selling_profit(entry_price, current_price):
        return ((entry_price - current_price) / entry_price) * 100

    actions = ["HOLD"] * (len(forecasted_prices)+1)
    holding_days = 0
    entry_price = None
    current_profit = None

    # Calculate the percentage changes for each day based on the forecast
    forecasted_changes = (forecasted_prices - current_price) / current_price * 100
    # Step 1: Check if forecasted profit is >= 5% for the first day

    if any(forecasted_changes <= -5):
        entry_price = current_price
        actions[0] = "BORROW-SELL"
    else:
        return actions

    # Continue checking for the next 9 days
    for day in range(0, len(forecasted_prices)):
        holding_days += 1
        current_price = forecasted_prices[day]
        current_profit = calculate_short_selling_profit(entry_price, current_price)  # Short position profit

        if current_profit >= 5 and day < len(forecasted_prices) - 1:
            # Check if profit < 5% on subsequent days

            tomorrows_profit = calculate_short_selling_profit(entry_price, forecasted_prices[day + 1])

            is_tomorrows_profit_less_than_5_percent_from_current = tomorrows_profit < current_profit + 5

            if is_tomorrows_profit_less_than_5_percent_from_current:
                actions[day] = "BUY-RETURN"
                return actions
        elif current_profit < 0:  # Current loss
            if forecasted_changes[day] > forecasted_changes[day - 1]:
                # Tomorrow's loss is expected to be greater than today's -> sell (short selling)
                actions[day] = "BUY-RETURN"
                return actions

    actions[0] = "BUY-RETURN"

    return actions

def get_share_count_to_borrow(current_price, initial_money):
    # in this strategy we borrow stock that have a value of our initial money
    # initial_money /= 2 # but could be half of the initial money
    return int(initial_money / current_price)