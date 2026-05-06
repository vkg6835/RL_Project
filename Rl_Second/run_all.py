import subprocess
import sys


def main():
    cmd = [
        sys.executable,
        "research_experiment.py",
        "--data_root",
        "data/Metacsv",
        "--eeg_root",
        "data/data_preprocessed_python",
        "--output_dir",
        "output",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
