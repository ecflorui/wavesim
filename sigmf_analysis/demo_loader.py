"""Task 1 demo: print the first record's shapes and metadata.

    python demo_loader.py data/v3_g16_mcs9_classical.zip
"""
import argparse

from sigmf_loader import load_run, summarize

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run folder or .zip")
    a = ap.parse_args()
    recs = load_run(a.run)
    print(f"{len(recs)} records loaded\n")
    print(summarize(recs[0]))
