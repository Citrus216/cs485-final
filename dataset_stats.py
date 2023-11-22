from sys import argv
import os
import csv

def main(args):
    """
    This script is used to collect statistics about the dataset. Statistics collected include:\n
    - Number of documents\n
    - Number of words across all documents\n
    - Average number of words per document\n
    - Data file formats\n
    - Provided dataset names\n
    - Number of documents per dataset\n
    - Number of words per dataset\n
    - Average number of words per document per dataset\n
    - Average number of annotations per document per dataset\n
    :param args: output file path
    These statistics are dumped into a file with name designated by the user.
    :return: None
    """
    if len(args) != 2:
        print("usage: python dataset_stats.py <path>")
        exit(1)

    output_filename = args[1]

    doc_count = 0
    word_count = 0
    avg_word_count = 0
    data_file_formats = set()
    dataset_names = set()
    dataset_doc_counts = {}
    dataset_word_counts = {}
    dataset_avg_word_counts = {}
    dataset_annotation_counts = {}
    dataset_avg_annot_counts = {}

    files_seen: set = set()

    try:
        # get all tsv files in the given directory
        for dir in [f for f in os.listdir() if os.path.isdir(f)]:  # only look at directories
            # ignore hidden dirs
            if dir.startswith("."):
                continue

            dataset_names.add(dir)
            dataset_doc_counts[dir] = 0
            dataset_word_counts[dir] = 0
            dataset_avg_word_counts[dir] = 0
            dataset_annotation_counts[dir] = 0

            for file in os.listdir(dir):
                filename = file.split(".")[0]
                if filename not in files_seen:
                    data_file_formats.add(file.split(".")[-1])
                    files_seen.add(filename)
                    doc_count += 1
                    dataset_doc_counts[dir] += 1
                    if file.endswith(".txt"):
                        with open(os.path.join(dir, file), "r") as f:
                            text_len = len(f.read().split())

                            word_count += text_len
                            dataset_word_counts[dir] += text_len
                    elif file.endswith(".tsv"):
                        with open(os.path.join(dir, file), "r") as f:
                            reader = csv.reader(f, delimiter="\t")
                            dataset_annotation_counts[dir] += len(list(reader))

    except NotADirectoryError or NotImplementedError as e:
        print("Invalid path. Make sure it is a valid directory.")
        print(e)
        exit(1)
    except FileNotFoundError as e:
        print("File not found. Make sure the path is correct and that the file exists.")
        print(e)
        exit(1)

    avg_word_count = word_count / doc_count
    for dir in dataset_names:
        dataset_avg_word_counts[dir] = dataset_word_counts[dir] / dataset_doc_counts[dir]
        dataset_avg_annot_counts[dir] = dataset_annotation_counts[dir] / dataset_doc_counts[dir]

    with open(output_filename, "w") as f:
        f.write("Number of documents: " + str(doc_count) + "\n")
        f.write("Number of words across all documents: " + str(word_count) + "\n")
        f.write("Average number of words per document: " + str(avg_word_count) + "\n")
        f.write("Data file formats: " + str(data_file_formats) + "\n")
        f.write("Provided dataset names: " + str(dataset_names) + "\n")
        f.write("Number of documents per dataset: " + str(dataset_doc_counts) + "\n")
        f.write("Number of words per dataset: " + str(dataset_word_counts) + "\n")
        f.write("Average number of words per document per dataset: " + str(dataset_avg_word_counts) + "\n")
        f.write("Average number of annotations per document per dataset: " + str(dataset_avg_annot_counts) + "\n")


if __name__ == "__main__":
    main(argv)
