from datasets import load_dataset

dataset = load_dataset(
    "winterForestStump/10-K_sec_filings"
)

row = dataset["001"][0]

for key, value in row.items():

    if value is None:
        print(key, "-> NONE")

    else:
        print(key, "->", len(str(value)))