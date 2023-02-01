# adapted from pettingzoo.classic.connect_four
import os
import pygame

def get_image(path):

    cwd = os.path.dirname(__file__)
    image = pygame.image.load(os.path.join(cwd, path))
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc

def load_chip(tile_size, filename, scale):
    chip = get_image(os.path.join("img", filename))
    chip = pygame.transform.scale(
        chip, (int(tile_size * (scale)), int(tile_size * (scale)))
    )
    return chip

def load_chip_preview(tile_size, filename, scale):
    chip = get_image(os.path.join(os.path.join("img","preview"), filename))
    chip = pygame.transform.scale(
        chip, (int(tile_size * (scale)), int(tile_size * (scale)))
    )
    return chip

# from https://github.com/michaelfeil/skyjo_rl/blob/dev/rlskyjo/utils.py
from pathlib import Path
from typing import Union
import glob
import os
import re

def get_project_root() -> Path:
    """return Path to the project directory, top folder of rlskyjo
    Returns:
        Path: Path to the project directory
    """
    return Path(__file__).parent.parent.resolve()

def find_file_in_subdir(parent_dir: Union[Path, str], file_str: Union[Path, str], regex_match: str = None) -> Union[str, None]:
    files = glob.glob(
        os.path.join(parent_dir, "**", file_str), recursive=True
    )
    if regex_match is not None:
        p = re.compile(regex_match)
        files = [ s for s in files if p.match(s) ]
    return sorted(files)[-1] if len(files) else None

# adapted from https://commons.wikimedia.org/wiki/File:SquareWaveFourierArrows.gif#Source_code
import time
import subprocess

class GIFRecorder:
    """
        This class is used to record a PyGame surface and save it to a gif file.
    """
    def __init__(self, width, height, out_file=f'game.gif'):
        """
        Initialize the recorder with parameters of the surface.
        :param width: Width of the surface to capture
        :param height: Height of the surface to capture
        :param fps: Frames per second
        :param out_file: Output file to save the recording
        """
        print(f'Initializing GifWriter with parameters width:{width} height:{height}')
        print(f'Output of the recording will be saved to {out_file}.')
        self.filename_list = []
        self.frame_num = 0
        self.start_time = time.time()
        self.path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir), os.pardir) # Root project directoy
        print(self.path)
        self.out_file = out_file
        self.ended = False



    def capture_frame(self, surf):
        """
         Call this method every frame, pass in the pygame surface to capture.
        :param surf: pygame surface to capture
        :return: None
        """
        """

            Note: surface must have the dimensions specified in the constructor.
        """
        # transform the pixels to the format used by open-cv
        self.filename_list.append(os.path.join(self.path, f'temp_{time.time()}_' + str(self.frame_num) + '.png'))
        pygame.image.save(surf, self.filename_list[-1])
        self.frame_num += 1

    # Convert indivual image files to GIF
    def end_recording(self):
        """
        Call this method to stop recording.
        :return: None
        """
        if not self.ended:
            # stop recording
            duration = time.time() - self.start_time
            seconds_per_frame = duration/ self.frame_num
            frame_delay = str(int(seconds_per_frame * 100))
            command_list = ['convert', '-delay', frame_delay, '-loop', '0'] + self.filename_list + [self.out_file]
            # Use the "convert" command (part of ImageMagick) to build the animation
            subprocess.call(command_list, cwd=self.path)
            # Earlier, we saved an image file for each frame of the animation. Now
            # that the animation is assembled, we can (optionally) delete those files
            for filename in self.filename_list:
                os.remove(filename)
            print(f"Saved recording to {self.out_file}")
            self.ended = True
