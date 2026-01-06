class YieldCurveBuilder:
    """Constructs yield curves for selected dates."""

    def __init__(self, market_data):
        self.market_data = market_data

    def build_curve(self, date):
        curve = self.market_data.get_curve_for_date(date)
        return curve.squeeze()
