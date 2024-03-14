import matplotlib.pyplot as plt
import shutil
import os

def plot_images(input_image, similar_images, number_retrieval, path):
    f, axes = plt.subplots(5,2, figsize=(15, 15))

    # Plot input image
    axes[2, 0].imshow(input_image)
    axes[2, 0].set_title('Input Image')
    axes[2, 0].axis('off')

    # Plot similar images
    for i, image in enumerate(similar_images):
        axes[i, 1].imshow(image)
        axes[i, 1].axis('off')

    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)
    axes[3, 0].set_visible(False)
    axes[4, 0].set_visible(False)
    
    plt.tight_layout()
    # plt.show()

    f.savefig(f'{path}/retrieval_{number_retrieval}.png')
    plt.close(f)

def reset_folder(path):
    try:
        shutil.rmtree(path)

    except:
        pass

    os.mkdir(path)