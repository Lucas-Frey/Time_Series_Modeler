from Base.BaseModeler import BaseModeler

import pmdarima as pm
import warnings

class AutoArimaModeler(BaseModeler):

    def __init__(self, multiseries: bool = False, parallelize: bool = False):

        super(AutoArimaModeler, self).__init__(multiseries, parallelize)

    def fit(self, df):
        pass

    def predict(self, df):
        pass

    def fit_predict(self, df):
        y_col_name = 'demand'

        n_diffs = 0
        ns_diffs = 0
        test_m = 7
        m = 1

        adf_d = pm.arima.utils.ndiffs(df[y_col_name], test='adf')
        kpss_d = pm.arima.utils.ndiffs(df[y_col_name], test='kpss')
        pp_d = pm.arima.utils.ndiffs(df[y_col_name], test='pp')
        n_diffs = int(max(adf_d, kpss_d, pp_d))

        # Code for seasonality tests.

        ch_D = pm.arima.utils.nsdiffs(df[y_col_name], m=7, test='ch')
        ocsb_D = pm.arima.utils.nsdiffs(df[y_col_name], m=7, test='ocsb')
        ns_diffs = int(max(ch_D, ocsb_D))

        if ns_diffs == 0:
            m = 1
        else:
            m = test_m

        # Code for seasonality tests.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = pm.auto_arima(df[y_col_name],
                                  m=m,
                                  d=n_diffs,
                                  D=ns_diffs,
                                  max_order=8,
                                  error_action='ignore',
                                  trace=False,
                                  information_criterion='bic',
                                  sarimax_kwargs={'enforce_invertibility': True})

        predictions = model.predict(n_periods=56)

        return predictions
