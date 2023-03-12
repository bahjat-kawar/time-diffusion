import numpy as np
import pandas as pd

test_set = []

def populate_test_set(begin_idx = 0, end_idx = 104, dataset_fname="TIMED_test_set_filtered_SD14.csv"):
    df = pd.read_csv(dataset_name)
    for idx in range(begin_idx, end_idx):
        entry = {}
        entry["old"] = df["old"][idx]
        entry["new"] = df["new"][idx]
        entry["positives"] = []
        entry["negatives"] = []
        for i in range(1, 6):
            entry["positives"].append({
                "test": df[f"positive{i}"][idx],
                "gt": df[f"gt{i}"][idx]
            })
            entry["negatives"].append({
                "test": df[f"negative{i}"][idx],
                "gt": df[f"gn{i}"][idx]
            })
        test_set.append(entry)