import matplotlib.pyplot as plt
import shutil
import os

translator = {
    0: 'coast', 
    1: 'forest', 
    2: 'highway', 
    3: 'insidecity', 
    4: 'mountain', 
    5: 'Opencountry', 
    6: 'street', 
    7: 'tallbuilding'
}

def plot_images(query_image, query_index, query_label, similar_images, images_labels, path):
    f, axes = plt.subplots(5,2, figsize=(15, 15))

    # Plot input image
    axes[2, 0].imshow(query_image)
    axes[2, 0].set_title(f'{translator[query_label]}')
    axes[2, 0].axis('off')

    # Plot similar images
    for i, image in enumerate(similar_images):
        axes[i, 1].imshow(image)
        axes[i, 1].set_title(f'{translator[images_labels[i]]}')
        axes[i, 1].axis('off')

    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)
    axes[3, 0].set_visible(False)
    axes[4, 0].set_visible(False)
    
    plt.tight_layout()
    # plt.show()

    f.savefig(f'{path}/retrieval_{query_index}.png')
    plt.close(f)

def reset_folder(path):
    try:
        shutil.rmtree(path)

    except:
        pass

    os.mkdir(path)