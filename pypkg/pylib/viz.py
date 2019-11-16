import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def plot_calibration_curve(y_test, prob_pos, output=".", name="PetSet Model", show=False):
    # Based on https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name,))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output, 'Calibration Curve.png'))
    # plt.savefig(os.path.join(output, 'Calibration Curve.pdf'))
    if show:
        plt.show()
    plt.close(fig)


def display(display_list, title=None, save_fn=None, one_per_window=True):
    fig = plt.figure(figsize=(15, 15))
    total_display = len(display_list)
    items = len(display_list[0])
    for i, display_item in enumerate(display_list):
        if one_per_window:
            plt.subplot(1, total_display, i + 1)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_item))
            if title:
                plt.title(title[i])
        else:
            for j, img in enumerate(display_item):
                fig.add_subplot(total_display, items, i * items + j + 1)
                plt.imshow(tf.keras.preprocessing.image.array_to_img(img))
                if title and i == 0:
                    plt.title(title[j])
        plt.axis('off')
    if save_fn:
        plt.savefig(save_fn)
    else:
        plt.show()
