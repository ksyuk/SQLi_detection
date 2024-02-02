import matplotlib.pyplot as plt


def plot_loss(report_count, loss_values):
    plt.plot(report_count, loss_values, color='dimgray')
    plt.xlabel('Count')
    plt.ylabel('Loss')
    plt.show()


def plot_inference_time_histogram(inference_time, bins='auto'):
    plt.hist(inference_time, bins=bins, color='dimgray')
    plt.xlabel('Inference Time')
    plt.ylabel('Count')
    plt.show()
