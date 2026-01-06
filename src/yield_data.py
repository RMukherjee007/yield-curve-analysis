import pandas as pd


class YieldMarketData:
    """Loads and stores G-Sec yield data across maturities."""

    def __init__(self, csv_file):
        self.data = self._load_data(csv_file)

    def _load_data(self, csv_file):
        df = pd.read_csv(csv_file, parse_dates=['Date'])
        return df.sort_values('Date')

    def get_curve_for_date(self, date):
        date = pd.to_datetime(date)
        row = self.data[self.data['Date'] == date]
        return row.drop(columns=['Date']) if not row.empty else pd.DataFrame()
