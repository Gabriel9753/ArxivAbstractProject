import json
import os

import pandas as pd
import toml
from icecream import ic

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

ic(f"Script directory: {script_dir}")

# Load configuration
config_path = os.path.join(script_dir, "preprocess_config.toml")
config = toml.load(config_path)

json_path = os.path.join(script_dir, config["paths"]["json_path"])
parquet_path = os.path.join(script_dir, config["paths"]["parquet_path"])
sample_parquet_path = os.path.join(script_dir, config["paths"]["sample_parquet_path"])
category_map_path = os.path.join(script_dir, config["paths"]["category_map_path"])
load_existing = config["settings"]["load_existing"]
chunksize = config["settings"]["chunksize"]
title_word_quantiles = config["settings"]["title_word_quantiles"]
abstract_word_quantiles = config["settings"]["abstract_word_quantiles"]
lb_papers = config["settings"]["lb_papers"]
n_samples = config["settings"]["n_samples"]
categories_of_interest = config["settings"]["categories_of_interest"]

# Ensure cache directory exists
cache_dir = os.path.join(script_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)


def load_data(json_path, parquet_path, load_existing):
    """Loads data from a parquet file if exists, otherwise from JSON and saves it to parquet."""
    if load_existing and os.path.exists(parquet_path):
        ic(f"Loading existing parquet file from {parquet_path}")
        return pd.read_parquet(parquet_path, engine="pyarrow")
    else:
        return load_json_to_dataframe(json_path, parquet_path)


def load_json_to_dataframe(json_path, parquet_path):
    """Loads data from a JSON file into a DataFrame and saves it to a parquet file."""
    columns_to_keep = ["id", "title", "abstract", "categories", "update_date"]
    dtypes = {col: str for col in columns_to_keep}
    reader = pd.read_json(json_path, lines=True, chunksize=chunksize)
    data_to_keep = []

    for i, chunk in enumerate(reader):
        ic(f"Processing chunk {i}")
        data_to_keep.append(chunk[columns_to_keep].to_dict(orient="records"))

    data_to_keep = [item for sublist in data_to_keep for item in sublist]
    df = pd.DataFrame(data_to_keep).reset_index(drop=True)
    for col in columns_to_keep:
        df[col] = df[col].astype(dtypes[col])
    df.to_parquet(parquet_path, compression="gzip", engine="pyarrow")
    return df


def preprocess_data(df, category_map_path):
    """Preprocesses the DataFrame by adding new columns and mapping categories."""
    with open(category_map_path, "r") as f:
        category_map = json.load(f)

    df["title_words"] = df["title"].apply(lambda x: len(x.split()))
    df["abstract_words"] = df["abstract"].apply(lambda x: len(x.split()))
    df["categories"] = df["categories"].apply(lambda x: x.split())
    df["mapped_categories"] = df["categories"].apply(lambda x: [category_map.get(cat, "UNKNOWN") for cat in x])
    df["amount_categories"] = df["categories"].apply(lambda x: len(x))
    df["update_year"] = df["update_date"].apply(lambda x: int(x[:4]))

    return df


def remove_duplicate_categories(df):
    """Removes duplicate categories from the DataFrame."""
    df["mapped_categories"] = df["mapped_categories"].apply(lambda x: list(set(x)))
    return df


def filter_papers_by_category_count(df, min_count=1, max_count=1):
    """Filters papers based on the number of categories they belong to."""
    return df[(df["amount_categories"] >= min_count) & (df["amount_categories"] <= max_count)].reset_index(drop=True)


def filter_papers_by_word_quantiles(df, title_quantiles, abstract_quantiles):
    """Filters papers based on word count quantiles for titles and abstracts."""

    title_quantiles_values = df["title_words"].quantile(0.1), df["title_words"].quantile(0.75)
    abstract_quantiles_values = df["abstract_words"].quantile(0.25), df["abstract_words"].quantile(0.75)

    return df[
        (df["title_words"] >= title_quantiles_values[0])
        & (df["title_words"] <= title_quantiles_values[1])
        & (df["abstract_words"] >= abstract_quantiles_values[0])
        & (df["abstract_words"] <= abstract_quantiles_values[1])
    ].reset_index(drop=True)


def filter_papers_by_category_frequency(df, lb_papers):
    """Filters papers by category frequency, retaining only those with a minimum number of papers."""
    category_counts = df["mapped_categories"].explode().value_counts().sort_values(ascending=False)
    return df[df["mapped_categories"].apply(lambda x: category_counts[x[0]] >= lb_papers)].reset_index(drop=True)


def sample_papers(df, n_samples, categories_of_interest):
    """Samples a specified number of papers, optionally filtered by categories of interest."""
    if categories_of_interest:
        df = df[df["mapped_categories"].apply(lambda x: x[0] in categories_of_interest)]
    return df.sample(n_samples, random_state=42).reset_index(drop=True)


def main():
    data_df = load_data(json_path, parquet_path, load_existing)
    ic(f"Loaded papers: {len(data_df)}")
    data_df = preprocess_data(data_df, category_map_path)
    ic(f"Preprocessed papers: {len(data_df)}")
    data_df = remove_duplicate_categories(data_df)
    ic(f"Removed duplicate categories: {len(data_df)}")
    data_df = filter_papers_by_category_count(data_df)
    ic(f"Filtered papers by category count: {len(data_df)}")
    data_df = filter_papers_by_word_quantiles(data_df, title_word_quantiles, abstract_word_quantiles)
    ic(f"Filtered papers by word quantiles: {len(data_df)}")
    data_df = filter_papers_by_category_frequency(data_df, lb_papers)
    ic(f"Filtered papers by category frequency: {len(data_df)}")
    sampled_df = sample_papers(data_df, n_samples, categories_of_interest)
    ic(f"Sampled papers: {len(sampled_df)}")
    sampled_df.to_parquet(sample_parquet_path, compression="gzip", engine="pyarrow")
    ic(f"Sampled papers: {len(sampled_df)}")


if __name__ == "__main__":
    main()
