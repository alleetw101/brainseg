import argparse


def parsercli():
    parser = argparse.ArgumentParser(prog='brainseg')

    parser.add_argument('integer', type=int, help='Integer to cub')
    parser.add_argument('--power', '-p', type=int, help='Integer to nth power')

    args = parser.parse_args()

    if args.power:
        print(args.integer**args.power)
    else:
        print(args.integer**3)


def main():
    parsercli()
