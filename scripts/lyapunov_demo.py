#!/usr/bin/env python
import argparse
from ebm.model.jepa import SudokuJEPA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    model = SudokuJEPA.load_from_checkpoint(args.checkpoint)
    model.eval()

    print("✅ Lyapunov extension ready")

if __name__ == "__main__":
    main()
