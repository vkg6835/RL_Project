import argparse
import subprocess
import sys

from music_emotion_rl import SARSAAgent, ExpectedSARSAAgent


def main():
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for the new SARSA pipeline."
    )
    parser.add_argument("--subject_id", type=int, default=1)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument(
        "--algorithm",
        choices=["sarsa", "expected_sarsa"],
        default="sarsa",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "music_emotion_pipeline.py",
        "--output_dir",
        args.output_dir,
        "--subject_id",
        str(args.subject_id),
        "--trial",
        str(args.trial),
        "--algorithm",
        args.algorithm,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
