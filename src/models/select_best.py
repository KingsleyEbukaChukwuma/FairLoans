import pandas as pd


def select_best(results):

    df = pd.DataFrame(results)

    df = df.sort_values(

        "ROC AUC",

        ascending=False

    )

    return df.iloc[0]