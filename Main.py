from M5.SalesTrainValidationReader import SalesTrainValidationReader
from M5.AutoArimaModeler import AutoArimaModeler
from Utils.SerialPrinter import SerialPrinter
import pandas as pd

def main():

    printer = SerialPrinter()
    printer.print_notification('Beginning M5 competition program...')
    reader = SalesTrainValidationReader('Data/sales_train_validation.csv')
    df = reader.read()

    modeler = AutoArimaModeler(multiseries=True, parallelize=True)

    predictions_df = modeler.multi_fit_predict(df)

    cols = ['F_' + str(i) for i in range(1, 57)]
    cols.insert(0, 'id')

    predictions_df.index.rename('id', inplace=True)
    predictions_df.reset_index(drop=False, inplace=True)
    predictions_df.columns = cols

    val_cols = ['F_' + str(i) for i in range(1, 29)]
    val_cols.insert(0, 'id')
    eval_cols = ['F_' + str(i) for i in range(29, 57)]
    eval_cols.insert(0, 'id')

    predictions_df_val = predictions_df.loc[:, val_cols]
    predictions_df_val['id'] = predictions_df_val['id'].apply(lambda x: str(x) + '_validation')

    predictions_df_eval = predictions_df.loc[:, eval_cols]
    predictions_df_eval['id'] = predictions_df_eval['id'].apply(lambda x: str(x) + '_evaluation')
    predictions_df_eval.columns = val_cols

    submission_df = pd.concat([predictions_df_val, predictions_df_eval])

    submission_df.to_csv('submission.csv', index=False)

    i = 0


if __name__ == "__main__":
    main()