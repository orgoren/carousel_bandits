import json
import argparse

def most_frequent(lst):
    return max(set(lst), key=lst.count)

def main(args):
    with open(args.results_file) as f:
        d = json.load(f)

    minimums = []

    regrets = {}
    for policy in d:
        regrets[policy] = [d[policy][0]]
        for i in range(1, len(d[policy])):
            regrets[policy].append(d[policy][i] - d[policy][i-1])

    for i in range(100):
        minimum = 1000000000000
        minimum_policy = "NA"
        for policy in d:
            if regrets[policy][i] < minimum:
                minimum = regrets[policy][i]
                minimum_policy = policy
        minimums.append(minimum_policy)

    minimums_range = {}

    for i in range(0,len(minimums), 10):
        minimums_range[i] = most_frequent(minimums[i:i+10])
    import pprint

    pprint.pprint(minimums_range)
    print("good")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-results_file", default="results.json")
    return parser.parse_args()

if __name__ == "__main__":
    with open("exp4_stats.json") as f:
        d = json.load(f)

    print("good")
    #main(get_arguments())
