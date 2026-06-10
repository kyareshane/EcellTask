import pandas as pd

df = pd.read_csv(
    "data/processed/dataset.csv"
)

q1 = df["length"].quantile(0.33)
q2 = df["length"].quantile(0.66)

def label(length):

    if length < q1:
        return 0

    elif length < q2:
        return 1

    else:
        return 2

df["label"] = df["length"].apply(label)

print(df["label"].value_counts())

df.to_csv(
    "data/processed/labeled_dataset.csv",
    index=False
)