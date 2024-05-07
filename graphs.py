import os
import logging
import yaml
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(level=logging.WARNING)


def compare_accuracy(data: dict, criterion: str):
    """
    Compare the accuracy of the model based on the criterion provided.
    :param data: dictionary containing the data from the YAML files
    :param criterion: the criterion to compare the model accuracy on
    :return: None
    """

    model_names = 'distilbert'
    collators = 'custom'
    reframings = 'reframed'
    base = f"{model_names}_{collators}_{reframings}"

    if criterion == 'model':
        model_names = 'bert'
        label_base = 'on DistilBERT'
        label_tocomp = 'on BERT'
    elif criterion == 'collator':
        collators = 'default'
        label_base = 'with custom masking'
        label_tocomp = 'with default masking'
    elif criterion == 'reframing':
        reframings = 'original'
        label_base = 'with reframing'
        label_tocomp = 'without reframing'
    else:
        logging.error(f'{criterion} is not a valid criterion.')
        raise ValueError('Invalid criterion')

    tocompare = f"{model_names}_{collators}_{reframings}"

    sindig_base_acc = data[base]['single-digit_accuracy']
    sindig_base_acc = [x[0] for x in sindig_base_acc]
    doubdig_base_acc = data[base]['double-digit_accuracy']
    doubdig_base_acc = [x[0] for x in doubdig_base_acc]
    singdig_tocompare_acc = data[tocompare]['single-digit_accuracy']
    singdig_tocompare_acc = [x[0] for x in singdig_tocompare_acc]
    doubdig_tocompare_acc = data[tocompare]['double-digit_accuracy']
    doubdig_tocompare_acc = [x[0] for x in doubdig_tocompare_acc]

    n_epochs = len(sindig_base_acc)

    plt.style.use('ggplot')

    # add alpha channel to the color to make it lighter
    plt.figure(facecolor='white')

    # use more gray ticks for vertical axis
    plt.yticks(np.arange(0, 101, 10))

    # use more ticks for horizontal axis
    plt.xticks(np.arange(0, n_epochs+1))

    # chnage grid color to light gray
    plt.grid(color='#e9e9e9')

    # use lighter background for the plot
    plt.gca().set_facecolor('#f9f9f9')

    epochs = np.arange(0, n_epochs)

    plt.plot(
        epochs, sindig_base_acc,
        label=f'Single-digit {label_base}',
        color="blue", alpha=0.5
        )
    plt.plot(
        epochs, singdig_tocompare_acc,
        label=f'Single-digit {label_tocomp}',
        color="red", alpha=0.5
        )
    plt.plot(
        epochs, doubdig_base_acc,
        label=f'Double-digit {label_base}',
        color="lightblue", alpha=0.9
        )
    plt.plot(
        epochs, doubdig_tocompare_acc,
        label=f'Double-digit {label_tocomp}',
        color="lightcoral", alpha=0.7
        )

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('')  # Impact of Numerical Reframing on Model Accuracy
    plt.legend()

    # save the plot
    plt.savefig(f'../results/{criterion}_comparison.png')


def main():
    files = os.listdir('../results')
    data = {}
    for file in files:
        parts = file.split('_')
        model_name = parts[0].split('-')[0]
        collator = parts[1]
        reframing = parts[3]
        if file.endswith('.yaml'):
            with open('../results/' + file, 'r') as f:
                data_from_file = yaml.load(f, Loader=yaml.FullLoader)
                data[f"{model_name}_{collator}_{reframing}"] = data_from_file

    compare_accuracy(data, 'model')
    compare_accuracy(data, 'collator')
    compare_accuracy(data, 'reframing')


if __name__ == '__main__':
    main()
