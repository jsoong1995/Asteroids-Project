from pygame.math import Vector2
from pygame.transform import rotozoom

from utils import get_random_velocity, load_sprite, wrap_position

UP = Vector2(0, -1)
RIGHT = Vector2(1, 0) # Eric Added

class GameObject:
    def __init__(self, position, sprite, velocity):
        self.position = Vector2(position)
        self.sprite = sprite
        self.radius = sprite.get_width() / 2
        self.velocity = Vector2(velocity)

    def draw(self, surface):
        blit_position = self.position - Vector2(self.radius)
        surface.blit(self.sprite, blit_position)

    def move(self, surface, ship_pos, frame): # Added Extra Inputs (but they're not used)
        self.position = wrap_position(self.position + self.velocity, surface)

    def collides_with(self, other_obj):
        distance = self.position.distance_to(other_obj.position)
        return distance < self.radius + other_obj.radius

class Spaceship(GameObject):
    MANEUVERABILITY = 3 # Degrees to Rotate Velocity by Each Frame
    ACCELERATION = 0.25 # Factor to Increase Velocity by Each Frame
    DECELERATION = -0.25 # Factor to Decrease Velocity by Each Frame
    BULLET_SPEED = 3 # Magnitude of Bullet Speed

    def __init__(self, position, create_bullet_callback):
        self.create_bullet_callback = create_bullet_callback
        # Make a copy of the original UP vector
        self.direction = Vector2(RIGHT) # Eric Modified

        super().__init__(position, load_sprite("spaceship"), Vector2(0))

    def accelerate(self):
        self.velocity += self.direction * self.ACCELERATION

    def decelerate(self):
        self.velocity += self.direction * self.DECELERATION

    def rotate(self, clockwise=True):
        sign = 1 if clockwise else -1
        angle = self.MANEUVERABILITY * sign
        self.direction.rotate_ip(angle)

    def draw(self, surface):
        angle = self.direction.angle_to(UP)
        rotated_surface = rotozoom(self.sprite, angle, 1.0)
        rotated_surface_size = Vector2(rotated_surface.get_size())
        blit_position = self.position - rotated_surface_size * 0.5
        surface.blit(rotated_surface, blit_position)

    # Added Move Function Here to Overide the Main One. Takes position from CSV file
    def move(self, surface, ship_pos, frame):
        # self.position = wrap_position(self.position + self.velocity, surface)
        self.position = wrap_position(ship_pos[frame], surface)

    def shoot(self):
        bullet_velocity = self.direction * self.BULLET_SPEED + self.velocity
        bullet = Bullet(self.position, bullet_velocity)
        self.create_bullet_callback(bullet)

class Asteroid(GameObject):
    def __init__(self, position, create_asteroid_callback, size=3):
        self.create_asteroid_callback = create_asteroid_callback
        self.size = size

        size_to_scale = {
            3: 1,
            2: 0.5,
            1: 0.25,
        }
        scale = size_to_scale[size]
        sprite = rotozoom(load_sprite("asteroid"), 0, scale)

        velocity = get_random_velocity(1, 3) # Eric Added

        print(f'Asteroid Start Location Vector: {position}') # Eric Added
        print(f'Asteroid Start Velocity Vector: {velocity}') # Eric Added

        super().__init__(
            position, sprite, velocity # Eric Modified
        )

    def split(self):
        if self.size > 1:
            for _ in range(2):
                asteroid = Asteroid(
                    self.position, self.create_asteroid_callback, self.size - 1
                )
                self.create_asteroid_callback(asteroid)

class Bullet(GameObject):
    def __init__(self, position, velocity):
        super().__init__(position, load_sprite("bullet"), velocity)

    def move(self, surface):
        self.position = self.position + self.velocity

class Goal(GameObject):
    def __init__(self,position):
        self.direction = Vector2(UP)
        self.size = 20
        super().__init__(position,load_sprite("goal"),(0,0))

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
