import argparse, json
import pandas as pd
from .config import LEAGUES, DEFAULT_SEASONS, PROCESSED_DIR
from .data import download_dataset, load_raw_matches
from .features import build_features
from .modeling import train_models

def main():
    parser = argparse.ArgumentParser(description="Soccer Predict Lab v2")
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--leagues", nargs="+", default=list(LEAGUES))
    d.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    sub.add_parser("prepare")
    sub.add_parser("train")
    sub.add_parser("build-all")
    args = parser.parse_args()

    if args.cmd=="download":
        for p in download_dataset(args.leagues,args.seasons):
            print("Downloaded",p)
    elif args.cmd=="prepare":
        df = build_features(load_raw_matches())
        PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
        df.to_csv(PROCESSED_DIR/"features.csv",index=False)
        print("Saved",len(df),"rows")
    elif args.cmd=="train":
        df = pd.read_csv(PROCESSED_DIR/"features.csv",parse_dates=["Date"])
        print(json.dumps(train_models(df),indent=2))
    elif args.cmd=="build-all":
        download_dataset(list(LEAGUES),DEFAULT_SEASONS)
        df = build_features(load_raw_matches())
        PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
        df.to_csv(PROCESSED_DIR/"features.csv",index=False)
        print(json.dumps(train_models(df),indent=2))

if __name__=="__main__":
    main()
