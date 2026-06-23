"""Command line entrypoint for video QA data generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from video_qa.config import CONFIG
from video_qa.pipeline.generator import VideoQAGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=CONFIG.cli.description)
    parser.add_argument(CONFIG.cli.input_dir_arg, type=Path, default=Path(CONFIG.cli.input_dir_default))
    parser.add_argument(CONFIG.cli.output_dir_arg, type=Path, default=CONFIG.output.output_dir)
    parser.add_argument(
        CONFIG.cli.max_workers_arg,
        type=int,
        default=CONFIG.pipeline.default_max_workers,
        help="Maximum number of video processes running concurrently.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = VideoQAGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )
    generator.run()


if __name__ == "__main__":
    main()
