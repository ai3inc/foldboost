import os
import sys


def main(argv):
    flags = {}
    for i, t in enumerate(argv[:-1]):
        if t.startswith("--"):
            flags[t] = argv[i + 1]
    print("fakepkg ran", argv, flush=True)
    if "--out" in flags:
        with open(flags["--out"], "w") as f:
            f.write(str(os.getpid()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
