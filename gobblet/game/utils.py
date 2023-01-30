import os
import pygame

# Modified from pettingzoo.classic.connect_four
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