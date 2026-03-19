from sklearn.utils import resample
import pandas as pd

def balance_dataset(df):

    hate = df[df['class'] == 0]
    non_hate = df[df['class'] == 1]

    hate_upsampled = resample(
        hate,
        replace=True,
        n_samples=len(non_hate),
        random_state=42
    )

    df_balanced = pd.concat([hate_upsampled, non_hate])

    return df_balanced