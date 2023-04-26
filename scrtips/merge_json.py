# python merge_json.py a.json b.json c.json
# a.json + b.json -> c.json

import argparse
import json

def merge_json_files(input_files, output_file):
    result = list()
    for f1 in input_files:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    with open(output_file, 'w') as output_file:
        json.dump(result, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge multiple JSON files into one.')
    parser.add_argument('input_files', metavar='input_file', type=str, nargs='+',
                        help='one or more JSON input files to merge')
    parser.add_argument('output_file', metavar='output_file', type=str,
                        help='output file to save the merged JSON data')

    args = parser.parse_args()
    merge_json_files(args.input_files, args.output_file)
