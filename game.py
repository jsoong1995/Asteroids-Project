import pygame

from models import Asteroid, Spaceship, Goal
from utils import get_random_position, get_random_start_position, load_sprite
from planning import motion_planning

import csv

from pygame.math import Vector2

'''
Changelog

4/17/2021
1) Added self.goal
2) modified build_cspace() with goal handling

To Do: Figure out how to pass the numpy arrays from planning.py into rrt_3D.py


'''


class Asteroids:

    MIN_ASTEROID_DISTANCE = 250

    def __init__(self):
        self._init_pygame()
        screen_width = 800 # Eric Added
        screen_height = 600 # Eric Added
        self.screen = pygame.display.set_mode((screen_width, screen_height)) # Eric Modified
        # self.background = load_sprite("space", False)
        self.clock = pygame.time.Clock()

        self.fps = 30 # Eric Added

        self.asteroids = []
        self.bullets = []

        # self.spaceship = Spaceship(get_random_start_position(self.screen), self.bullets.append) # Eric Modified
        self.spaceship = Spaceship(Vector2(100,200), self.bullets.append) # Eric Modified
        self.goal = Goal((600,450))

        print(f'Spacecraft Initial Location X: {self.spaceship.position[0]} Y: {self.spaceship.position[1]}')
        print(f'Goal Location X: {self.goal.position[0]} Y: {self.goal.position[1]}')

        num_asteroids = 0 # Eric Added

        for _ in range(num_asteroids): # Eric Modified
            while True:
                position = get_random_position(self.screen)
                if (
                    position.distance_to(self.spaceship.position)
                    > self.MIN_ASTEROID_DISTANCE
                ):
                    break

            self.asteroids.append(Asteroid(position, self.asteroids.append))

        motion_planning(self.spaceship,self.goal,self.asteroids,self.fps,screen_width,screen_height)
        # Eric Added
        # John added self.goal

    def main_loop(self):
        frame = 0
        ship_pos = []
        with open('results.txt') as motionPlan:
            spaceshipCoords = csv.reader(motionPlan, delimiter=',')
            for row in spaceshipCoords:
                ship_pos.append(Vector2(float(row[0]),float(row[1])))

        while True:
            self._handle_input()
            self._process_game_logic(ship_pos,frame)
            self._draw()
            frame += 1

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Asteroids Recreation - ENAE788V")

    def _get_game_objects(self):
        game_objects = [*self.asteroids, *self.bullets]

        if self.spaceship:
            game_objects.append(self.spaceship)

        return game_objects

    def _handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                quit()
            elif (
                self.spaceship
                and event.type == pygame.KEYDOWN
                and event.key == pygame.K_SPACE
            ):
                self.spaceship.shoot()

        is_key_pressed = pygame.key.get_pressed()

        if self.spaceship:
            if is_key_pressed[pygame.K_RIGHT]:
                self.spaceship.rotate(clockwise=True)
            elif is_key_pressed[pygame.K_LEFT]:
                self.spaceship.rotate(clockwise=False)
            if is_key_pressed[pygame.K_UP]:
                self.spaceship.accelerate()
            elif is_key_pressed[pygame.K_DOWN]:
                self.spaceship.decelerate()

    def _process_game_logic(self,ship_pos,frame):
        for game_object in self._get_game_objects():
            game_object.move(self.screen,ship_pos,frame)

        if self.spaceship:
            for asteroid in self.asteroids:
                if asteroid.collides_with(self.spaceship):
                    self.spaceship = None
                    break

        for bullet in self.bullets[:]:
            for asteroid in self.asteroids[:]:
                if asteroid.collides_with(bullet):
                    self.asteroids.remove(asteroid)
                    self.bullets.remove(bullet)
                    asteroid.split()
                    break

        for bullet in self.bullets[:]:
            if not self.screen.get_rect().collidepoint(bullet.position):
                self.bullets.remove(bullet)

        if self.spaceship:
            if self.goal.collides_with(self.spaceship):
                self.spaceship = None


    def _draw(self):
        self.screen.fill((0, 0, 25))
        # self.screen.blit(self.background, (0, 0))

        for game_object in self._get_game_objects():
            game_object.draw(self.screen)
        # print("Collides:", self.spaceship.collides_with(self.asteroid))

        self.goal.draw(self.screen)

        pygame.display.flip()
        self.clock.tick(self.fps) # Eric Added

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
