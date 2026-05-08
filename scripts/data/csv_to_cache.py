# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#### motivated by this repo: https://github.com/microsoft/mattergen/blob/main/mattergen/scripts/csv_to_dataset.py

import argparse
import os

import torch

from scripts._bootstrap import set_default_env
set_default_env()

from dao.common.data_utils import preprocess


def convert_one_csv(
    file_path: str,
    save_path: str,
    *,
    niggli: bool,
    primitive: bool,
    graph_method: str,
    prop: str,
    num_workers: int = 30,
):
    """Convert a single CSV file to a `*_ori.pt` cache."""
    print(f"Processing {file_path}")
    processed_data = preprocess(
        input_file=file_path,
        niggli=niggli,
        primitive=primitive,
        graph_method=graph_method,
        prop_list=[prop],
        num_workers=num_workers,
    )
    torch.save(processed_data, save_path)


def convert_csv_folder(
    csv_folder: str,
    cache_path: str,
    *,
    niggli: bool,
    primitive: bool,
    graph_method: str,
    prop: str,
    num_workers: int = 30,
):
    """Convert all CSV files in a folder to `*_ori.pt` caches."""
    for file in os.listdir(f"{csv_folder}"):
        if not file.endswith(".csv"):
            continue
        file_path = os.path.join(csv_folder, file)
        save_path = cache_path or f"{file_path.split('.')[0]}_ori.pt"

        if os.path.exists(save_path):
            print(f"File {save_path} already exists. Skipping processing.")
            continue

        convert_one_csv(
            file_path,
            save_path,
            niggli=niggli,
            primitive=primitive,
            graph_method=graph_method,
            prop=prop,
            num_workers=num_workers,
        )


def main():
    parser = argparse.ArgumentParser(description="Convert CSV files to PyTorch cache format.")
    
    parser.add_argument(
        "--csv_folder",
        type=str,
        required=True,
        help="Path to the folder containing the csv files. All csv files in the folder will be processed (e.g., 'train.csv', 'val.csv', 'test.csv') and the resulting datasets will be placed under {cache_path/dataset_name/filename_without_extension}, e.g, /path/to/project/dataset/mp_20/train.",
    )
    
    parser.add_argument(
        "--cache_path",
        type=str,
        default="",
        help="Path to the cache folder where the processed datasets will be saved.",
    )
    
    parser.add_argument(
        "--niggli",
        default=True,
        type=bool,
        help="Whether to use Niggli reduction for the crystal structure.",
    )
    
    parser.add_argument(
        "--primitive",
        default=False,
        type=bool,
        help="Whether to use primitive cell for the crystal structure.",
    )
    
    parser.add_argument(
        "--graph-method",
        type=str,
        default="crystalnn",
        help="Method to convert crystal structure to graph.",
    )
    
    parser.add_argument(
        "--prop",
        type=str,
        default="ehull",
        help="Property to be extracted from the csv files.",
    )

    
    args = parser.parse_args()

    convert_csv_folder(
        args.csv_folder,
        args.cache_path,
        niggli=args.niggli,
        primitive=args.primitive,
        graph_method=args.graph_method,
        prop=args.prop,
        num_workers=30,
    )

if __name__ == "__main__":
    main()
