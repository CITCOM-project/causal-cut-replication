"""
This module concatenates separate dataframes representing individual runs into one single dataframe in "long format" for
survival analysis.
"""

from glob import glob
import os
import sys
import pandas as pd


assert len(sys.argv) == 2, "Please provide a directory of csv files."
ROOT = sys.argv[1]

chunk_headers = {}
header = True

if not os.path.exists(f"{ROOT}_attacks"):
    os.mkdir(f"{ROOT}_attacks")

for run_id in sorted(glob(f"{ROOT}/*.pqt")):
    if "data" in os.path.basename(run_id):
        continue
    df = pd.read_parquet(run_id)
    (chunk_id,) = set(df["attack_inx"])

    if "id" not in df:
        print(df)
    df["id"] = os.path.basename(run_id).split(".")[0]
    df["Blood_Glucose"] = df.pop("Blood Glucose")
    df["Blood_Insulin"] = df.pop("Blood Insulin")
    df["time"] = df.pop("step")

    df.to_csv(
        f"{ROOT}_attacks/{chunk_id}.csv",
        mode="w" if chunk_headers.get(chunk_id, True) else "a",
        header=chunk_headers.get(chunk_id, True),
        index=False,
    )
    # df.to_csv(f"{ROOT}/data.csv", mode="w" if header else "a", header=header, index=False)
    header = False
    chunk_headers[chunk_id] = False

for chunk in glob(f"{ROOT}_attacks/*.csv"):
    pd.read_csv(chunk).to_parquet(chunk.replace(".csv", ".pqt"))
    os.remove(chunk)
