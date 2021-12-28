import random

# Use this to generate SAME asteroid starting location and velocity repeatedly
# Comment it out to fully "randomize" asteroid location and velocity
# random.seed(3) # Eric Added

from pygame.image import load
from pygame.math import Vector2

def load_sprite(name, with_alpha=True):
    path = f"assets/sprites/{name}.png"
    loaded_sprite = load(path)

    if with_alpha:
        return loaded_sprite.convert_alpha()
    else:
        return loaded_sprite.convert()

def wrap_position(position, surface):
    x, y = position
    w, h = surface.get_size()
    return Vector2(x % w, y % h)

def get_random_position(surface):
    return Vector2(
        random.randrange(surface.get_width()),
        random.randrange(surface.get_height()),
    )

def get_random_start_position(surface): # Eric Added Function
    return Vector2(
        random.randrange(0.25*surface.get_width()), # Eric Modified 0.25
        random.randrange(surface.get_height()),
    )

def get_random_velocity(min_speed, max_speed):
    speed = random.randint(min_speed, max_speed)
    angle = random.randrange(0, 360)
    return Vector2(speed, 0).rotate(angle)

'''
Asteroids Simulation Developed Following "Build and Asteroids Game With Python
and Pygame" by Pawel Fertyk

Full Tutorial Found at: https://realpython.com/asteroids-game-python/

Tutorial Citation:
Fertyk, P. (2021, March 22). Build an Asteroids Game With Python and Pygame. RealPython.com.
    https://realpython.com/asteroids-game-python/

Full Source Code Found at: https://github.com/realpython/materials/tree/master/asterioids-pygame-project

Source Code Citation:
Amos, D. (2021). Build an Asteroids Game With Python and Pygame. RealPython.
    https://github.com/realpython/materials/tree/master/asterioids-pygame-project
'''
