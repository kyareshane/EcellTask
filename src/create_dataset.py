from datasets import load_dataset
import pandas as pd

dataset = load_dataset(
    "winterForestStump/10-K_sec_filings"
)

rows = []

MAX_RECORDS = 5000
count = 0

for split_name in dataset.keys():

    split = dataset[split_name]

    for row in split:

        business = str(row["Business"])

        mda = str(
            row["Management’s Discussion and Analysis of Financial Condition and Results of Operations"]
        )

        text = business + " " + mda

        rows.append({
            "text": text
        })

        count += 1

        if count >= MAX_RECORDS:
            break

    if count >= MAX_RECORDS:
        break

df = pd.DataFrame(rows)

df["length"] = df["text"].str.len()

print(df["length"].describe())

df.to_csv(
    "data/processed/dataset.csv",
    index=False
)