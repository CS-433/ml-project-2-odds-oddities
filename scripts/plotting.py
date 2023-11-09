"""TODO: add description."""
from matplotlib import pyplot as plt


def show_images(axis=True, tight_layout=False, **images):
    image_count = len(images)
    plt.figure(figsize=(image_count * 3, 3))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, image_count, i + 1)
        plt.axis('off') if not axis else None
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=14)
        plt.imshow(image, cmap="Greys_r")
    plt.tight_layout() if tight_layout else None
    plt.show()
