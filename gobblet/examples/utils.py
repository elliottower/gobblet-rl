# adapted from https://commons.wikimedia.org/wiki/File:SquareWaveFourierArrows.gif#Source_code
import pygame
import time
import subprocess
import os

class GIFRecorder:
    """
        This class is used to record a PyGame surface and save it to a gif file.
    """
    def __init__(self, width, height, out_file=f'anim_{time.time()}.gif'):
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
        self.path = os.path.dirname(os.path.realpath(__file__))
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