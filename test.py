
import argparse

parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("--a")
args = parser.parse_args()
print(args.a)
