import torch
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import argparse
from typing import Union, Literal, Iterable
import csv


def main(args):
    parser = argparse.ArgumentParser(description='Extract flags from command line arguments')
    parser.add_argument('-g', '--gpu', action='store_true', help='Attempt to enable GPU usage')
    parsed_args = parser.parse_args(args)
    use_gpu = parsed_args.gpu

    if use_gpu:
        # Check the GPU is detected
        if not torch.cuda.is_available():
            print("ERROR: No GPU detected. Please add a GPU; if you're using Colab, use their UI.")
            assert False
        # Get the GPU device name.
        device_name = torch.cuda.get_device_name()
        n_gpu = torch.cuda.device_count()
        print("Found device: {}, n_gpu: {}".format(device_name, n_gpu))
    else:
        # Check that no GPU is detected
        if torch.cuda.is_available():
            print("ERROR: GPU detected.")
            print("Remove the GPU or set the use_gpu flag to True.")
            assert False
        print("No GPU found. Using CPU.")
        print("WARNING: Without a GPU, your code will be extremely slow.")

    pretrained_bert = 'bert-base-uncased'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_bert)
    model = BertForSequenceClassification.from_pretrained(pretrained_bert, output_hidden_states=True).to(device)

    training_data = get_dataset('train')
    test_data = get_dataset('test')
    finetune_dataset = get_dataset('dev')

    training_features = extract_bert_features(model, tokenizer, [sentence for sentence, _ in training_data])
    training_labels = [labels for _, labels in training_data]

    logreg_models = []
    # train and test an LR model for each label
    for i in range(len(training_data[0][1])):
        cur_labels = [label[i] for label in training_labels]
        logreg_model = LogisticRegression(max_iter=10000).fit(training_features, cur_labels)
        logreg_models.append(logreg_model)

    # train and test a single NN with one output for each label type and 1 hidden layer of 128 neurons and 768 inputs
    neural_net = MLPClassifier(hidden_layer_sizes=(128,), max_iter=10000, activation='logistic').fit(training_features, training_labels)

    test_features = extract_bert_features(model, tokenizer, [sentence for sentence, _ in test_data])
    test_labels = [labels for _, labels in test_data]

    # test
    logreg_performance = test_logreg(logreg_models, test_features, test_labels)
    nn_performance = test_nn(neural_net, test_features, test_labels)

    # evaluate
    evaluate(logreg_performance)
    evaluate_nn(nn_performance)


def get_dataset(name: Union[Literal['train', 'dev', 'test']]) -> list[tuple[str, list[int]]]:
    filename = f"{name}_data_collapsed.csv"
    with open(filename, "r") as f:
        data = []
        reader = csv.reader(f)
        next(reader)
        for sentence, *labels in reader:
            data.append((sentence, list(map(lambda l: int(l), labels))))
        return data


def extract_bert_features(model, tokenizer, sentences):
    """
    Extracts BERT features from a list of sentences. We only want [CLS] token for the last layer.
    :param model:
    :param tokenizer:
    :param sentences:
    :return:
    """
    features = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            last_hidden_states = model(**inputs).hidden_states[-1]
            cls_token_values = last_hidden_states[:, 0, :].detach().numpy()
            features.append(cls_token_values[0])
    return features


def test_logreg(models, features, labels) -> dict[LogisticRegression, list[str]]:
    # test each model on the whole set of features and labels
    models_dict = {model: [] for model in models}
    for i, model in enumerate(models):
        samples = model.predict(features)
        true_labels = [label[i] for label in labels]
        models_dict[model] = (samples, true_labels)
    # label TP, TN, FP, FN per model
    performance = {}
    for model, (samples, true_labels) in models_dict.items():
        performance[model] = []
        for sample, label in list(zip(samples, true_labels)):
            if sample == 1 and label == 1:
                performance[model].append("TP")
            elif sample == 0 and label == 0:
                performance[model].append("TN")
            elif sample == 1 and label == 0:
                performance[model].append("FP")
            elif sample == 0 and label == 1:
                performance[model].append("FN")
            else:
                raise Exception("Something went wrong.")
    return performance

def test_nn(model, features, labels) -> list[float]:
    # test on test set
    samples = model.predict(features)
    # output fraction of correct labels per feature
    performance = []
    for sample_set, label_set in list(zip(samples, labels)):
        print(sample_set, label_set, "\n", sep="\n")
        num_correct = 0
        num_total = 0
        for i, label in enumerate(label_set):
            # ignore 0's for now
            if label == 1:
                if sample_set[i] == label:
                    num_correct += 1
                num_total += 1
        if num_total != 0:
            performance.append(num_correct / num_total)
        else:
            performance.append(1.0)
    return performance


def evaluate(performance: dict[Union[LogisticRegression, MLPClassifier], list[str]]):
    # print performance
    for i, (_, performance_vector) in enumerate(performance.items()):
        print(f"Model {i}:")
        print(f"TP: {performance_vector.count('TP')}")
        print(f"TN: {performance_vector.count('TN')}")
        print(f"FP: {performance_vector.count('FP')}")
        print(f"FN: {performance_vector.count('FN')}")
        print(f"Accuracy: {(performance_vector.count('TP') + performance_vector.count('TN')) / len(performance_vector)}")
        if performance_vector.count('TP') + performance_vector.count('FP') != 0:
            precision = performance_vector.count('TP') / (performance_vector.count('TP') + performance_vector.count('FP'))
        else:
            precision = 0
        print(f"Precision: {precision}")
        if performance_vector.count('TP') + performance_vector.count('FN') != 0:
            recall = performance_vector.count('TP') / (performance_vector.count('TP') + performance_vector.count('FN'))
        else:
            recall = 0
        print(f"Recall: {recall}")
        if precision + recall != 0:
            print(f"F1: {2 * precision * recall / (precision + recall)}")
        else:
            print(f"F1: 0")
        print()


def evaluate_nn(performance: list[float]):
    # print performance
    print("Neural Net:")
    print(f"Accuracy: {sum(performance) / len(performance)}")
    print(performance)
    print()


if __name__ == "__main__":
    from sys import argv
    main(argv[1:])
