# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:22:15 2024

@author: orhan
"""

###Reinforcement Learning 

import pygame 
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

#window size

WIDTH = 360
HEIGHT = 360
FPS = 500

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

class Player(pygame.sprite.Sprite):
    #sprite for the player 
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20)) #Agent
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, WHITE, self.rect.center, self.radius)
        self.rect.centerx = WIDTH/2
        self.rect.centery = HEIGHT - 10
        self.speedx = 0
        self.speedy = 0
        
        
    def update(self, action):
        self.speedx = 0
        self.speedy = 0
        keystate = pygame.key.get_pressed()
        
        if keystate[pygame.K_LEFT] or action == 0:
            self.speedx = -20
        elif keystate[pygame.K_RIGHT] or action == 1:
            self.speedx = 20
        elif keystate[pygame.K_DOWN] or action == 2:
            self.speedy = -20
        elif keystate[pygame.K_UP] or action == 3:
            self.speedy = 20
            
        else:
            self.speedx == 0
            
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT
            
            
    def getCoordinates(self):
        return(self.rect.x, self.rect.y)




class Enemy(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.rect.x = random.randrange(0, WIDTH - self.rect.width)
        self.rect.y = random.randrange(0, HEIGHT - self.rect.height)
        
        self.speedx = 10
        self.speedy = 10
        
        
    def update(self):
        
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        
        if self.rect.top > HEIGHT + 10:
            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
            self.rect.y = random.randrange(2,6)
            self.speedy = 10
            
    def getCoordinates(self):
        return(self.rect.x, self.rect.y)            
            

class Point(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, GREEN, self.rect.center, self.radius)
        self.rect.x = random.randrange(0, WIDTH - self.rect.width)
        self.rect.y = random.randrange(0, HEIGHT - self.rect.height)
        self.speedx = 0
        self.speedy = 0
        
    def update(self):
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        
        if self.rect.top > HEIGHT + 10:
            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
            self.rect.y = random.randrange(2,6)
            self.speedy = 0

    def getCoordinates(self):
        return(self.rect.x, self.rect.y)         

        
            
class DQLAgent:
    def __init__(self):
        #parameter
        self.state_size = 12 #distance
        self.action_size = 5 #right, left ,up, down, none
        
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 10000000)
        
        self.model = self.build_model()
        
        

    def build_model(self):
        #neural network for deep q learning
        
        model = Sequential()
        
        model.add(Dense(10,input_dim = self.state_size, activation ="relu"))
        model.add(Dense(self.action_size, activation = "linear"))
        model.compile(loss = "mse", optimizer= Adam(learning_rate = self.learning_rate))
        return model
    
    def remember(self,state,action,reward,next_state,done):
        
        #storage
        self.memory.append((state,action,reward,next_state,done))
        
    def act(self,state):
        
        #act exploer or exploit
        
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state,verbose = 0)
        return np.argmax(act_values[0])
        
    
    def replay(self,batch_size):
        #training / batch_size hafıza'dan kaç tane bilgi seçeceğimizi bize söyler.

        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory,batch_size)
        for state, action ,reward ,next_state, done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            
            if done:
                target = reward
            else:
                target = reward + self.gamma*np.amax(self.model.predict(next_state, verbose = 0))
            train_target = self.model.predict(state, verbose = 0)
            train_target[0][action] = target
            self.model.fit(state,train_target,verbose = 0)
    
    
    def adapativeEGreddy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
            
            
class Env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.points = pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1 = Enemy()
        self.m2 = Enemy()
        self.m3 = Enemy()
        self.m4 = Enemy()
        self.m5 = Enemy()
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.all_sprite.add(self.m3)
        self.all_sprite.add(self.m4)
        self.all_sprite.add(self.m5)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.enemy.add(self.m3)
        self.enemy.add(self.m4)
        self.enemy.add(self.m5)
        self.point = Point()
        self.all_sprite.add(self.point)
        self.points.add(self.point)
        
        
        self.reward = 0 
        self.total_reward = 0 
        self.done = False
        self.agent = DQLAgent()
        
        
    def findDistance(self, a, b):
        d = a-b
        return d
    
    def step(self,action):
        state_list = []
        
        #updates
        self.player.update(action)
        self.enemy.update()
        
        #get cordinates
        next_player_state = self.player.getCoordinates()
        next_m1_state = self.m1.getCoordinates()
        next_m2_state = self.m2.getCoordinates()
        next_m3_state = self.m3.getCoordinates()
        next_m4_state = self.m4.getCoordinates()
        next_m5_state = self.m5.getCoordinates()
        next_point_state = self.point.getCoordinates()

        


        #find distance 
        
        state_list.append(self.findDistance(next_player_state[0], next_m1_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m1_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m2_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m2_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m3_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m3_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m4_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m4_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m5_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m5_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_point_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_point_state[1]))


        
        return [state_list]


    #reset
    def initialState(self):
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.points = pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1 = Enemy()
        self.m2 = Enemy()
        self.m3 = Enemy()
        self.m4 = Enemy()
        self.m5 = Enemy()
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.all_sprite.add(self.m3)
        self.all_sprite.add(self.m4)
        self.all_sprite.add(self.m5)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.enemy.add(self.m3)
        self.enemy.add(self.m4)
        self.enemy.add(self.m5)
        self.point = Point()
        self.all_sprite.add(self.point)
        self.points.add(self.point)
        
        
        self.reward = 0 
        self.total_reward = 0 
        self.done = False
        
        state_list = []
        
        #get coordinate 
        player_state = self.player.getCoordinates()
        m1_state = self.m1.getCoordinates()
        m2_state = self.m2.getCoordinates()
        m3_state = self.m3.getCoordinates()
        m4_state = self.m4.getCoordinates()
        m5_state = self.m5.getCoordinates()
        point_state = self.point.getCoordinates()
        
        
        
        # find distance                
        state_list.append(self.findDistance(player_state[0], m1_state[0]))
        state_list.append(self.findDistance(player_state[1], m1_state[1]))
        state_list.append(self.findDistance(player_state[0], m2_state[0]))
        state_list.append(self.findDistance(player_state[1], m2_state[1]))
        state_list.append(self.findDistance(player_state[0], m3_state[0]))
        state_list.append(self.findDistance(player_state[1], m3_state[1]))
        state_list.append(self.findDistance(player_state[0], m4_state[0]))
        state_list.append(self.findDistance(player_state[1], m4_state[1]))
        state_list.append(self.findDistance(player_state[0], m5_state[0]))
        state_list.append(self.findDistance(player_state[1], m5_state[1]))
        state_list.append(self.findDistance(player_state[0], point_state[0]))
        state_list.append(self.findDistance(player_state[1], point_state[1]))


        return [state_list]
    
    
    
    
    def run(self):
        #game loop
        
        state = self.initialState()
        running = True
        batch_size = 0
        while running:
            self.reward = 10
            
            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            #update
            action = self.agent.act(state)
            next_state = self.step(action)
            self.total_reward += self.reward
            
            hits = pygame.sprite.spritecollide(self.player,self.enemy,False , pygame.sprite.collide_circle)
            if hits:
                self.reward = -150
                self.total_reward += self.reward
                self.done = False
                running = False
                print("Total reward: ",self.total_reward)
                
            touch = pygame.sprite.spritecollide(self.player,self.points, False , pygame.sprite.collide_circle)
            if touch:
                self.reward = 100
                self.total_reward += self.reward
                self.done = True
                running = True
                print("Total reward: ",self.total_reward)
                                
                
                
            #storage
            self.agent.remember(state, action, self.reward, next_state, self.done)
            
            #update
            state = next_state 
            
            #training 
            self.agent.replay(batch_size)
            
            #epsilon
            self.agent.adapativeEGreddy()
            
            #draw
            screen.fill(BLACK)
            self.all_sprite.draw(screen)
            #after drawing flip display
            pygame.display.flip()
            
            
                
    pygame.quit()

if __name__ =="__main__":
    env = Env()
    liste = []
    t = 0
    while True:
        t += 1 
        print("Episode: ",t)
        liste.append(env.total_reward)
        
        #initilaize pygame and create window
        pygame.init()
        screen = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("RL Game")
        clock = pygame.time.Clock()
        
        env.run()
        
    
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        