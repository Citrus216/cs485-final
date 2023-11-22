from sys import argv
import csv
import os
from typing import Union, Literal, Iterable
import re

_files = {}

# These are the labels we want to use for our classification. The labels are taken from the original data set.
_desired_labels = [
    ["Name_Calling,Labeling"],
    ["Exaggeration,Minimisation"],
    ["Doubt", "Appeal_to_fear-prejudice"],
    ["Black-and-White_Fallacy"],
    ["Straw_Men", "Whataboutism", "Red_Herring"],
    ["Bandwagon"],
    ["Flag-Waving"],
    ["Repetition", "Slogans"]
]


def main(args):
    """
    Main function of the program. Takes a path as an argument and extracts data from all tsv files in that directory.
    :param args: path to the directory containing the tsv files
    :return: None
    """
    if len(args) != 2:
        print("usage: python data_extractor.py <path>")
        exit(1)

    doc_data: list = []

    dir_name = args[1]

    try:
        # get all tsv files in the given directory
        for file in os.listdir(dir_name):
            if file.endswith(".tsv"):
                # extract id from file name. using this re is dangerous if names are not consistent
                doc_data.extend(process_file(dir_name, file))
    except NotADirectoryError or NotImplementedError as e:
        print("Invalid path. Make sure it is a valid directory.")
        print(e)
        exit(1)
    except FileNotFoundError as e:
        print("File not found. Make sure the path is correct and that the file exists.")
        print(e)
        exit(1)

    # write data to file
    file_name_prefix = dir_name.strip('/') + ('_' if len(dir_name) > 0 else '')

    with open(f"{file_name_prefix}data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["document_id", "label", "text"])
        writer.writerows(doc_data)

    # collect labels
    # collect_labels(doc_data)

    # use desired labels to create a "collapsed" dataset where each line has the text and then several 0 or 1 flags for
    # each label group
    collapsed_data = {}
    for _, label, text in doc_data:
        if text not in collapsed_data:
            collapsed_data[text] = [0] * len(_desired_labels)
        for i, label_group in enumerate(_desired_labels):
            if label in label_group:
                collapsed_data[text][i] = 1

    with open(f"{file_name_prefix}data_collapsed.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["text"] + [str(label_group) for label_group in _desired_labels])
        writer.writerows([[text] + flags for text, flags in collapsed_data.items()])


def process_file(dir_name: str, file: str) -> list:
    """
    Extracts data from given file.
    :param file: name of the file from which to extract data
    :return: processed data as a list of lists. This can be considered tabulated data, as in a csv file.
    """
    ret = []

    full_name = f"{dir_name.rstrip('/') + '/'}{file}"

    with open(full_name, "r") as f:
        print("Processing " + file)
        reader = csv.reader(f, delimiter="\t" if ".tsv" in file else ",")
        for document_id, label, start, end, *_ in reader:
            ret.append((document_id, label, get_text(dir_name, document_id, int(start), int(end))))

    return ret


def get_text(dir_name: str, document_id: str, start: int, end: int) -> str:
    if document_id not in _files:
        full_name = f"{dir_name.rstrip('/') + '/'}article{document_id}.txt"
        # just open 2 for now so we can try the other encoding on error
        with open(full_name, "r", encoding="utf-8") as f_utf_8, open(full_name, "r", encoding="utf-16") as f_utf_16:
            try:
                _files[document_id] = f_utf_8.read()
            except UnicodeDecodeError:
                _files[document_id] = f_utf_16.read()

    return read_file(_files[document_id], start, end)


def read_file(content: str, start: int, end: int):
    # Define a regex pattern for sentence-ending punctuation
    sentence_end_pattern = re.compile(r'[.?!]\s*|\n')

    # Find the start and end indices of the sentence containing the specified span
    start_sentence_index = max(0, max(
        [match.end() for match in sentence_end_pattern.finditer(content, 0, start)] or [0]))

    end_sentence_index = sentence_end_pattern.search(content, end)

    if end_sentence_index:
        end_sentence_index = end_sentence_index.end()
    else:
        # If the end of the text is reached, consider the rest of the content as the sentence
        end_sentence_index = len(content)

    # Extract the sentence or set of sentences
    sentence = content[start_sentence_index:end_sentence_index].strip()
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    # replace quotes at the beginning and end with angled quotes
    sentence = re.sub(r'^"', '“', sentence)
    sentence = re.sub(r'"$', '”', sentence)

    return sentence


def collect_labels(csv_iterable: Iterable) -> set[str]:
    """
    Collects all labels from the data and writes them to a file.
    :return: None
    """
    labels = set()

    for _, label, _ in csv_iterable:
        labels.add(label)

    with open("labels.txt", "w") as f:
        f.write("\n".join(labels))

    return labels


if __name__ == "__main__":
    main(argv)
