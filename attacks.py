from matplotlib._api import itertools
from bthowen import WiSARD, binarize_datasets
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import ctypes as c
import seaborn as sns
import random
import torch
import wisardpkg as wp
import pandas as pd
from tqdm import tqdm


def get_datasets():
    # print("Loading dataset MNIST")
    train_dataset = dsets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    new_train_dataset = []
    for d in train_dataset:
        new_train_dataset.append((d[0].numpy().flatten(), d[1]))
    train_dataset = new_train_dataset
    test_dataset = dsets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )
    new_test_dataset = []
    for d in test_dataset:
        new_test_dataset.append((d[0].numpy().flatten(), d[1]))
    test_dataset = new_test_dataset

    return train_dataset, test_dataset


def add_noise(image, distortion):
    noisy_image = image + np.random.normal(0, distortion, 784)
    noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image


def plot_number(image):
    plt.imshow(1 - image.reshape(28, 28), cmap="gray")


def get_noisy_dataset(train_dataset, dataset, distortion, bits_per_input):
    # inicializa o dataset noisy
    noisy_dataset = []
    for d in dataset:
        noisy_dataset.append((add_noise(d[0], distortion), d[1]))

    X_noisy = []
    for d in noisy_dataset:
        X_noisy.append(d[0])

    X_noisy = np.array(X_noisy)

    # computa a média e o desvio padrão das entradas de treino
    std_skews = [
        norm.ppf((i + 1) / (bits_per_input + 1)) for i in range(bits_per_input)
    ]

    train_inputs = []

    for d in train_dataset:
        train_inputs.append(d[0])

    train_inputs = np.array(train_inputs)

    mean_inputs = train_inputs.mean(axis=0)
    std_inputs = train_inputs.std(axis=0)

    X_bin_noisy = []
    for i in std_skews:
        X_bin_noisy.append(
            (X_noisy >= mean_inputs + (i * std_inputs)).astype(c.c_ubyte)
        )

    X_bin_noisy = np.concatenate(X_bin_noisy, axis=1)

    return X_bin_noisy


def train_model(model, X_bin_train, y_train):
    for d in range(len(X_bin_train)):
        model.train(X_bin_train[d], y_train[d])
        # if ((d + 1) % 10000) == 0:
        #     print(d + 1)


def find_best_bleach(X_bin_val, y_val, model):
    # achando o melhor valor de bleaching com o dataset de validação

    max_val = 0
    for d in model.discriminators:
        for f in d.filters:
            max_val = max(max_val, f.data.max())
    # print(f"Maximum possible bleach value is {max_val}")
    # Use a binary search-based strategy to find the value of b that maximizes accuracy on the validation set
    best_bleach = max_val // 2
    step = max(max_val // 4, 1)
    bleach_accuracies = {}
    while True:
        values = [best_bleach - step, best_bleach, best_bleach + step]
        accuracies = []
        for b in values:
            if b in bleach_accuracies:
                accuracies.append(bleach_accuracies[b])
            elif b < 1:
                accuracies.append(0)
            else:
                pred = run_inference(X_bin_val, y_val, model, b)
                accuracy = 0
                for i, p in enumerate(pred):
                    if p == y_val[i]:
                        accuracy += 1
                bleach_accuracies[b] = accuracy
                accuracies.append(accuracy)
        new_best_bleach = values[accuracies.index(max(accuracies))]
        if (new_best_bleach == best_bleach) and (step == 1):
            break
        best_bleach = new_best_bleach
        if step > 1:
            step //= 2
    # print(f"Best bleach: {best_bleach}\n")
    return best_bleach


def run_inference(inputs, labels, model, bleach=1):
    num_samples = len(inputs)
    correct = 0
    ties = 0
    model.set_bleaching(bleach)
    pred = []
    for d in range(num_samples):
        prediction = model.predict(inputs[d])
        label = labels[d]
        if len(prediction) > 1:
            ties += 1
        prediction = np.random.choice(prediction)
        pred.append(prediction)
        if prediction == label:
            correct += 1
    # correct_percent = round((100 * correct) / num_samples, 4)
    # tie_percent = round((100 * ties) / num_samples, 4)
    # print(
    #     f"With bleaching={bleach}, accuracy={correct}/{num_samples} ({correct_percent}%); ties={ties}/{num_samples} ({tie_percent}%)"
    # )
    return pred


def run_experiment_bthowen(
    variant, noise, bleach=None, label=None, noise_train=0.0, with_confusion=False
):
    train_dataset, test_dataset = get_datasets()

    if noise_train > 0:
        replace_size = int(noise_train * len(train_dataset))
        replace_indices = np.random.choice(
            len(train_dataset), replace_size, replace=False
        )

        for i in replace_indices:
            train_dataset[i] = (
                add_noise(train_dataset[i][0], noise),
                train_dataset[i][1],
            )

    model_variants = {
        "small": {
            "bits_per_input": 2,
            "bits_per_filter": 28,
            "entries_per_filter": 1024,
            "hashes_per_filter": 2,
        },
        "large": {
            "bits_per_input": 6,
            "bits_per_filter": 49,
            "entries_per_filter": 8192,
            "hashes_per_filter": 4,
        },
    }
    parameters = model_variants[variant]

    X_bin_train, y_train, X_bin_val, y_val, X_bin_test, y_test = binarize_datasets(
        train_dataset, test_dataset, parameters["bits_per_input"]
    )

    X_bin_noisy = get_noisy_dataset(
        train_dataset, test_dataset, noise, parameters["bits_per_input"]
    )

    model = WiSARD(
        X_bin_train[0].size,
        y_train.max() + 1,
        parameters["bits_per_filter"],
        parameters["entries_per_filter"],
        parameters["hashes_per_filter"],
    )

    train_model(model, X_bin_train, y_train)

    best_bleach = bleach
    if bleach is None:
        best_bleach = find_best_bleach(X_bin_val, y_val, model)

    if label is not None:
        X_bin_test = X_bin_test[y_test == label]
        X_bin_noisy = X_bin_noisy[y_test == label]
        y_test = y_test[y_test == label]

    # print(f"\nExperimentos com {noise_train*100}% do treinamento ruidoso\n")
    # print("Experimento com 0 de ruído no dataset de teste:")
    pred = run_inference(X_bin_test, y_test, model, best_bleach)
    # print(f"Experimento com {noise} de ruído no dataset de teste:")
    pred_noisy = run_inference(X_bin_noisy, y_test, model, best_bleach)

    if with_confusion:
        conf_matrix, conf_percent = compute_confusion_matrix(pred, y_test)
        conf_matrix_noisy, conf_percent_noisy = compute_confusion_matrix(
            pred_noisy, y_test
        )

        save_figs(conf_matrix, conf_percent, 0, noise_train, f"bthowen_{variant}")
        save_figs(
            conf_matrix_noisy,
            conf_percent_noisy,
            noise,
            noise_train,
            f"bthowen_{variant}",
        )

    return (
        (np.array(pred) == np.array(y_test)).sum() / len(y_test),
        (np.array(pred_noisy) == np.array(y_test)).sum() / len(y_test),
    )


def run_experiment_wisard(variant, noise, noise_train=0.0, with_confusion=False):
    model_variants = {
        "small": {
            "bits_per_input": 2,
            "bits_address": 16,
        },
        "large": {
            "bits_per_input": 6,
            "bits_address": 32,
        },
    }
    parameters = model_variants[variant]

    train_dataset, test_dataset = get_datasets()

    if noise_train > 0:
        replace_size = int(noise_train * len(train_dataset))
        replace_indices = np.random.choice(
            len(train_dataset), replace_size, replace=False
        )

        for i in replace_indices:
            train_dataset[i] = (
                add_noise(train_dataset[i][0], noise),
                train_dataset[i][1],
            )

    X_bin_train, y_train, X_bin_val, y_val, X_bin_test, y_test = binarize_datasets(
        train_dataset, test_dataset, parameters["bits_per_input"]
    )

    X_bin_noisy = get_noisy_dataset(
        train_dataset, test_dataset, noise, parameters["bits_per_input"]
    )

    model = wp.Wisard(parameters["bits_address"])

    model.train(X_bin_train.tolist(), y_train.astype(str).tolist())

    # print(f"\nExperimentos com {noise_train*100}% do treinamento ruidoso\n")
    # print("Experimento com 0 de ruído no dataset de teste:")
    pred = model.classify(X_bin_test.tolist())
    correct = 0
    for i in range(len(pred)):
        correct += pred[i] == str(y_test[i])
    accuracy = correct / len(X_bin_test)
    # print(f"Acurácia: {accuracy}")

    # print(f"Experimento com {noise} de ruído no dataset de teste:")
    pred_noise = model.classify(X_bin_noisy.tolist())
    correct = 0
    for i in range(len(pred_noise)):
        correct += pred_noise[i] == str(y_test[i])
    accuracy_noise = correct / len(X_bin_noisy)
    # print(f"Acurácia: {accuracy_noise}")

    if with_confusion:
        conf_matrix, conf_percent = compute_confusion_matrix(pred, y_test)
        conf_matrix_noisy, conf_percent_noisy = compute_confusion_matrix(
            pred_noise, y_test
        )

        save_figs(conf_matrix, conf_percent, 0, noise_train, f"wisard_{variant}")
        save_figs(
            conf_matrix_noisy,
            conf_percent_noisy,
            noise,
            noise_train,
            f"wisard_{variant}",
        )

    return accuracy, accuracy_noise


def save_figs(conf_matrix, conf_matrix_percent, noise, noise_train, model_name):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt="g", cmap="Reds")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Absolute Values)")

    plt.subplot(1, 2, 2)
    sns.heatmap(conf_matrix_percent, annot=True, fmt=".1f", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Percentages)")

    plt.tight_layout()
    plt.savefig(f"images/{model_name}-noise_train={noise_train}-noise_test={noise}.png")
    plt.close()


def compute_confusion_matrix(pred, real):
    conf_matrix = np.zeros((10, 10), dtype=np.float64)

    for i in range(len(pred)):
        conf_matrix[int(real[i])][int(pred[i])] += 1

    conf_matrix_percent = (
        conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    )

    return conf_matrix, conf_matrix_percent


def main():
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    # duvidas
    # como escolher um wisard compativel com o bthowen small e large?
    # mesma coisa pra rede neural com peso

    # run_experiment_bthowen("small", noise=0.05, noise_train=0.05, with_confusion=True)
    # run_experiment_wisard("large", noise=0.001, noise_train=0, with_confusion=True)

    experiments = itertools.product(
        [("wisard", run_experiment_wisard), ("bthowen", run_experiment_bthowen)],
        ["small", "large"],  # variante dos modelos
        [0.001, 0.005, 0.01, 0.05],  # ruído adicionado
        [0, 0.1, 0.15, 0.2, 0.5, 1],  # porcentagem do dataset de treino com ruído
    )

    df = pd.DataFrame(
        columns=[
            "model",
            "variant",
            "noise",
            "noise_train",
            "accuracy",
            "accuracy_noise",
        ]
    )

    for (model, experiment), variant, noise, noise_train in tqdm(list(experiments)):
        accuracy, accuracy_noise = experiment(
            variant=variant, noise=noise, noise_train=noise_train, with_confusion=True
        )
        print(
            ["model", "variant", "noise", "noise_train", "accuracy", "accuracy_noise"]
        )
        print([model, variant, noise, noise_train, accuracy, accuracy_noise])
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [[model, variant, noise, noise_train, accuracy, accuracy_noise]],
                    columns=df.columns,
                ),
            ],
            ignore_index=True,
        )

    df.to_csv("results.csv", header=True, index=False)


if __name__ == "__main__":
    main()
