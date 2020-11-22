import os
import random

import cv2
import numpy as np
from tqdm import tqdm

CACHE_DIRECTORY = "cache"
SUTIABLE_IMG_THRESHOLD = 5

ATOM_HEIGHT_SCALE_DOWN = 120
ATOM_WIDTH_SCALE_DOWN = 120


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


class AtomImage:
    def __init__(self, filename, img_array, avg_color, shape, save_as_cache):
        self.filename = filename
        self.average_colour = avg_color


        if not img_array.shape == shape:
            self.image_array = cv2.resize(img_array, shape)
        else:
            self.image_array = img_array

        if save_as_cache:
            path = os.path.join(CACHE_DIRECTORY, filename)
            cv2.imwrite(path, self.image_array)

    @property
    def shape(self):
        return self.image_array.shape

    @classmethod
    def load_cached(cls, filename):
        image_array = cv2.imread(os.path.join(CACHE_DIRECTORY, filename), cv2.IMREAD_GRAYSCALE)

        average_colour = (sum([sum(x) / len(x) for x in image_array]) // len(image_array)).astype(np.uint8)

        return cls(filename, image_array, avg_color=average_colour, shape=image_array.shape, save_as_cache=False)


def is_cached(fp):
    return fp in os.listdir(CACHE_DIRECTORY)


class AtomImageProcessor:
    @staticmethod
    def process_images(directory, shape, cache: bool = False):
        print("Processing atom images...\n")
        if cache:
            assert os.path.exists(CACHE_DIRECTORY), "Cache directory does not exist"
            print("Caching images for further use...")

        assert os.path.exists(directory), f"{directory} directory does not exist"

        images = os.listdir(directory)
        atom_images = []
        assert len(images) > 1

        for image_path in tqdm(images):
            if cache and is_cached(image_path):
                atom_images.append(AtomImage.load_cached(image_path))
                continue

            img = cv2.imread(os.path.join(directory, image_path))
            monochrome = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if len(img) == 0:
                continue

            average_colour = (sum([sum(x) / len(x) for x in monochrome]) // len(monochrome)).astype(np.uint8)
            atom_images.append(
                AtomImage(filename=image_path, img_array=monochrome, avg_color=average_colour, shape=shape,
                          save_as_cache=cache))

        return atom_images


class TargetImage:
    def __init__(self, path, target_width):

        self.width = target_width
        self.path = path
        self.__processed = False

        image = cv2.imread(path)

        self.image = image_resize(image, width=self.width)
        self.tiles = self.process()

    @property
    def has_processed(self):
        return self.__processed

    @property
    def shape(self):
        return self.image.shape

    def process(self):
        print("Processing target image...")
        assert os.path.exists(self.path), "Target image does not exist"

        height, width, channels = shape = self.image.shape

        # Create cells

        tiles_dim = (height // ATOM_HEIGHT_SCALE_DOWN, width // ATOM_WIDTH_SCALE_DOWN)
        tiles = np.zeros((ATOM_HEIGHT_SCALE_DOWN, ATOM_WIDTH_SCALE_DOWN), dtype=np.uint8)

        for ny, y in enumerate(range(0, height, tiles_dim[0])):
            for nx, x in enumerate(range(0, width, tiles_dim[1])):
                region = self.image[y:y + tiles_dim[0], x:x + tiles_dim[1]]
                monochrome_image = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                average_colour = np.round(
                    sum([sum(x) / len(x) for x in monochrome_image]) / len(monochrome_image)).astype(np.uint8)
                tiles[ny][nx] = average_colour

        self.__processed = True
        return tiles


def pick_suitable_image(at_tile, atom_images):
    suitable_images = []

    best_score = 255

    for atom_img in atom_images:
        s = abs(atom_img.average_colour - at_tile)

        if s < best_score:
            best_score = s
            suitable_images.append(atom_img)

        if s < SUTIABLE_IMG_THRESHOLD:
            suitable_images.append(atom_img)

    return random.choice(suitable_images)


def stich_mosaic(mosaic, image, cx, cy):
    height, width = mosaic.shape
    atom_height, atom_width = image.shape

    #print(mosaic.shape, image.shape)
    #print(range(cx, cx+atom_width))

    for ny, y in enumerate(range(cy, cy+atom_height)):
        for nx, x in enumerate(range(cx, cx+atom_width)):
            mosaic[y][x] = image.image_array[ny][nx]

    return mosaic


def generate_mosaic(atom_images, target_image, atom_shape: tuple):
    assert target_image.has_processed

    mosaic_image = np.zeros(target_image.shape[:-1], dtype=np.uint8)

    print(mosaic_image.shape)

    atom_img_height, atom_img_width = atom_shape

    tiles_width, tiles_height = tiles_shape = target_image.tiles.shape

    height, width, _ = target_image.shape

    for ny, y in enumerate(range(0, height, atom_img_height)):
        for nx, x in enumerate(range(0, width, atom_img_width)):
            colour_at_tile = target_image.tiles[ny][nx]

            suit_image = pick_suitable_image(colour_at_tile, atom_images)

            mosaic_image = stich_mosaic(
                mosaic_image,
                suit_image,
                cx=x,
                cy=y,
            )

    return mosaic_image


def create_image(target_image_width, target_image_height, atom_img_dir="build", source_image="target.jpg", ):
    assert os.path.exists(atom_img_dir)
    assert os.path.exists(source_image)

    target_image_dir = source_image
    atom_images_dir = atom_img_dir

    atom_img_width = target_image_width // ATOM_WIDTH_SCALE_DOWN
    atom_img_height = target_image_height // ATOM_HEIGHT_SCALE_DOWN

    shape = (atom_img_height, atom_img_width)

    target_image = TargetImage("target.jpg", target_width=target_image_width)
    target_image.process()

    atom_images = AtomImageProcessor.process_images(atom_img_dir, (atom_img_width, atom_img_height), cache=True)

    mosaic_image = generate_mosaic(atom_images, target_image, atom_shape=shape)

    cv2.imwrite("output.jpg", mosaic_image)


if __name__ == "__main__":
    create_image(
        target_image_width=3840,
        target_image_height=2160
    )
