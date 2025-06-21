import argparse
import torch 
import pandas as pd
from tqdm import tqdm

def main(args):
    cached_data = torch.load(args.cache_path)
    # print(cached_data[0])
    cifs = []
    props = []
    ids = []

    for item in tqdm(cached_data):
        cifs.append(item['cif'])
        ids.append(item['mp_id'])
        props.append(item[args.prop_name])

    res = {
        'material_id': ids,
        'cif': cifs,
        f"{args.prop_name}": props
    }

    pd.DataFrame(res).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Convert cached data to CSV format.")

    argparser.add_argument(
        "--cache_path",
        type=str,
        default="/fs/fast/u2023000178/mattergen/datasets/pretrain_data/full.pt",
        help="Path to the cached data file.",
    )

    argparser.add_argument(
        "--output_csv",
        type=str,
        default="/fs/fast/u2023000178/mattergen/datasets/pretrain_data/full.csv",
        help="Path to the output CSV file.",
    )

    argparser.add_argument(
        "--prop_name",
        type=str,
        default="stability",
        help="Property name to be extracted from the cached data.",
    )
    
    args = argparser.parse_args()
    main(args)