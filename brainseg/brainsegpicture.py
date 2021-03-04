from PIL import Image
import numpy as np


class BrainSegPicture:

    def __init__(self, filepath):
        self.image = Image.open(filepath)

        if self.image.mode == 'RGBA':
            image = self.image.convert('LA')
        elif self.image.mode == 'RGB':
            image = self.image.convert('L')
        else:
            image = self.image

        image = np.array(image)
        self.original_image_array = image

        if image.ndim != 2:
            for column in range(len(image)):
                for row in range(len(image[0])):
                    if image[column][row][1] == 0:
                        image[column][row] = 0

            image = image[..., :1]

        self.image_array = image

    def information(self):
        print(self.original_image_array.shape)
        print(self.image_array.shape)
        print(self.image.mode)

    def save_image(self, filepath, channels):
        output_image = Image.fromarray(self.image_array)
        if channels == 3:
            output_image = output_image.convert('RGB')
        elif channels == 1:
            output_image = output_image.convert('L')
        else:
            TypeError(f'Number of channels {channels} not supported in output format.')

        output_image.save(filepath)