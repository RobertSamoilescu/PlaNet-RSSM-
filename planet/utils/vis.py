import cv2
import imageio
import numpy as np
from typing import List


def images_to_gif(
    observations: List[np.ndarray],
    filename: str,
    size=(256, 256),
    duration=2.0,
) -> None:
    """Save a list of observations as a GIF file.

    :param observations: List of observations
    :param filename: Name of the GIF file
    :param duration: Duration of each frame in seconds
    """
    images = []

    for image in observations:
        image = cv2.resize(image, size)
        images.append(image)

    # Write images to GIF using imageio
    imageio.mimsave(filename, images, format="GIF", duration=duration, loop=0)
