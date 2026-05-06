import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Read the saved RL comparison summary from the new pipeline."
    )
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()

    summary_path = os.path.join(args.output_dir, "rl", "rl_summary.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"{summary_path} was not found. Run research_experiment.py first."
        )

    summary = pd.read_csv(summary_path)
    print("=" * 72)
    print("Saved RL Comparison Summary")
    print("=" * 72)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
