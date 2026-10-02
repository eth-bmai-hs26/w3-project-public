import matplotlib.pyplot as plt


def _to_hwc(img):
    """Convert a (C, H, W) tensor to a numpy array ready for imshow.
    Returns (array, cmap) — cmap is 'gray' for grayscale, None for RGB.
    """
    img = img.cpu()
    if img.shape[0] == 1:
        return img.squeeze(0).numpy(), "gray"
    else:
        return img.permute(1, 2, 0).numpy(), None


def show_class_examples(dataset, class_names):
    """Show one example image for each class in the dataset."""
    num_classes = len(class_names)
    class_images = {}
    for img, lbl in dataset:
        if lbl not in class_images:
            class_images[lbl] = img
        if len(class_images) == num_classes:
            break

    fig, axes = plt.subplots(1, num_classes, figsize=(15, 1.8))
    for cls_idx, ax in enumerate(axes):
        img = (class_images[cls_idx] + 1) / 2
        img = img.clamp(0, 1)
        arr, cmap = _to_hwc(img)
        ax.imshow(arr, cmap=cmap)
        ax.set_title(class_names[cls_idx], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_images(images, titles=None, grid=False):
    """Display a batch of images, optionally as a grid."""
    images = (images + 1) / 2
    images = images.clamp(0, 1)

    if grid:
        cols = 4
        rows = (len(images) + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axs = axs.flatten() if rows > 1 else (axs if cols > 1 else [axs])
        for i, ax in enumerate(axs):
            if i < len(images):
                arr, cmap = _to_hwc(images[i])
                ax.imshow(arr, cmap=cmap)
                if titles and i < len(titles):
                    ax.set_title(titles[i])
            ax.axis('off')
    else:
        fig, axs = plt.subplots(1, len(images), figsize=(12, 3))
        for i, img in enumerate(images):
            arr, cmap = _to_hwc(img)
            axs[i].imshow(arr, cmap=cmap)
            axs[i].axis('off')
            if titles:
                axs[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()
