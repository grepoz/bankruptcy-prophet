from yahoofinancials import YahooFinancials

# https://finance.yahoo.com/quote/AAPL/financials?p=AAPL
# https://github.com/JECSand/yahoofinancials/tree/master


def get_quarter_data(financial_data, company, quarter_date):
    company_data = financial_data[company]
    for quarter in company_data:
        if quarter_date in quarter:
            return quarter[quarter_date]
    return {}


company = 'TSLA'
quarter_date = '2023-06-30'

yahoo_financials = YahooFinancials(company)

balance_sheet_data_qt = yahoo_financials.get_financial_stmts('quarterly', 'balance')
income_statement_data_qt = yahoo_financials.get_financial_stmts('quarterly', 'income')

try:
    balance_sheet_quarter_data = get_quarter_data(
        balance_sheet_data_qt['balanceSheetHistoryQuarterly'],
        company,
        quarter_date)

    income_statement_quarter_data = get_quarter_data(
        income_statement_data_qt['incomeStatementHistoryQuarterly'],
        company,
        quarter_date)

    working_capital = balance_sheet_quarter_data['workingCapital']
    retained_earnings = balance_sheet_quarter_data['retainedEarnings']
    earnings_before_interest_and_taxes = income_statement_quarter_data['ebit']
    total_liabilities = balance_sheet_quarter_data['totalLiabilitiesNetMinorityInterest']
    total_assets = balance_sheet_quarter_data['totalAssets']
    sales = income_statement_quarter_data['totalRevenue']

    market_value_of_equity = yahoo_financials.get_market_cap()

    altman_z_score = (1.2 * (working_capital / total_assets) +
                      1.4 * (retained_earnings / total_assets) +
                      3.3 * (earnings_before_interest_and_taxes / total_assets) +
                      0.6 * (market_value_of_equity / total_liabilities) +
                      1.0 * (sales / total_assets))

    print(altman_z_score)

except KeyError as e:
    print(f"KeyError occurred: {e}")

except Exception as e:
    print(f"Exception occurred: {e}")
