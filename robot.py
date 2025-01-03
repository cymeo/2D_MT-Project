import gymnasium as gym 
from gymnasium import spaces 
import numpy as np 
import math 
import matplotlib.pyplot as plt
from stable_baselines3 import PPO   
from stable_baselines3.common.monitor import Monitor


class TwoD_Robot(gym.Env): 
    
    def __init__(self):
        # stuff for training 
        self.max_episode_steps = 800
        self.current_episode = 0 
        self.current_step =0
        self.reward = 0
        self.total_rewards =0
        self.done = False
        # length of robot link    
        self.L1 = 42.5 
        self.L2 = 39.22 
        self.L3 = 9.97
        
        #initial robot motor angles and endeffektor pose  
        self.theta = np.array([0.0,0.0,0.0])
        self.ee_pose = self.forward_kinematics(self.theta)
        self.dist_to_goal = None
                
        # random goal initializeation
        self.goal = np.array([0,0])
        
        # Action space: motorangles 
        self.action_space = spaces.Box(low= -np.pi/2, high = np.pi/2, shape = (3,))
       
        # endeffektor pose, joint angles, goal, distance to goal space definition  
        self.observation_space = spaces.Dict(
            {
                "pose": spaces.Box(
                        low = np.array([-100,-100]),
                        high =np.array([100,100]), 
                        dtype = float
                    ), 
                "motors": spaces.Box(
                        low = np.array([-np.pi, -np.pi, -np.pi]),
                        high =np.array([ np.pi, np.pi, np.pi]),
                        dtype = float
                    ),
                 "goal": spaces.Box(
                        low = np.array([-100,-100]),
                        high =np.array([100,100]), 
                        dtype = float
                    ),
                "distance": spaces.Box(
                        low = np.array([0]),
                        high = np.array([200]),
                        dtype = float
                    )    
            } 
        )

        # Initialize plot
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 1)
        self.line1, = self.axes.plot([], [], '-o', label="Robot Joints")
        self.goal_marker, = self.axes.plot(0, 0, 'x', markersize=8, color='r', label="Goal")   
        self.axes.set_title("Robot Joint Positions")
        

    # function that returns ee_pose from given motorangles 
    def forward_kinematics(self,theta):
        """compute Endeffector Pose and returns x,y and theta (angles). zero pose is a vertical line"""
        x = (self.L1 * np.sin(theta[0]) +
            self.L2 * np.sin(theta[0] + theta[1]) +
            self.L3 * np.sin(theta[0] + theta[1] + theta[2]))
        y = (self.L1 * np.cos(theta[0]) +
            self.L2 * np.cos(theta[0] + theta[1]) +
            self.L3 * np.cos(theta[0] + theta[1] + theta[2]))
        #theta_E = theta[0] + theta[1] + theta[2]
        return np.array([x,y])
         
    # get observation / current state as dict with ee_pose, motorangles, goal, distance to goal 
    def _get_obs(self):
        self.ee_pose = self.forward_kinematics(self.theta)
        self.dist_to_goal = np.linalg.norm(self.ee_pose[:2] - self.goal[:2])
        dist_total = np.array([self.dist_to_goal])
    
        observation = {
            "pose": self.ee_pose,
            "motors": self.theta,
            "goal": self.goal, 
            "distance": dist_total
        }
       
        return observation
    
    #return empty dict
    def _get_info(self):
        info= {    
        } 
        return info   
    
    #return reward    
    def _get_reward(self):
        # distance reward
        reward = -self.dist_to_goal/100 *5
        # reard for goal  
        if self.done: 
            reward = 700
        #terminate after n steps                       
        if (self.current_step == self.max_episode_steps-1):
            reward = -700
        return reward        

    # set the robot to random state and defines new goal, returns new observation  and info(empty) 
    def reset(self, seed = None, options = None):
        #returns new random state and emptyinfo 
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.current_step =0

        #set new goal within radius of 60cm 
        rad = np.sqrt(np.random.uniform(0,1))* 60
        angle = np.random.uniform(-np.pi,np.pi)
        x = rad * np.cos(angle)
        y = rad * np.sin(angle)
        self.goal = np.array([x,y])
        
        #get observation and info
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
      
    def check_done (self):
        self.done = False
        reason= ''        
        if (self.dist_to_goal <= 5):
            self.done = True
            reason = 'goal reached'
        if (self.current_step>= self.max_episode_steps): 
            self.done = True
            reason = 'max_episodes reached' 
        
        if self.done: 
                self.current_episode +=1            
                #print(reason, self.current_episode, "steps: ", self.current_step)               
                self.current_step = 0 
                
        return self.done 
        
    #returns observation, reward, done = bool, truncated = False, info = empty  
    def step(self,action):
        # perform action and oberve 
        self.current_step +=1
        self.theta += action
        self.theta = np.clip(self.theta, -np.pi, np.pi)  # Keep angles in range
        self.ee_pose = self.forward_kinematics(self.theta)
    
        #observe
        observation = self._get_obs()
        done = self.check_done()
        self.reward = self._get_reward()
        truncated = False
        info = self._get_info()
        self.total_rewards+= self.reward          
        return observation, self.reward, done, truncated, info 
    
    # redenring without returning anything 
    def render(self, rendermode = "human"):
        if rendermode == 'human': 
           if self.done:
            angles = self.theta
            links = np.array([self.L1,self.L2,self.L3])
            #calc end of each Link
            Ey1 = math.cos(angles[0])*links[0]
            Ex1 = math.sin(angles[0])*links[0]
            Ey2 = math.cos(angles[0]+angles[1])*links[1] + Ey1
            Ex2 = math.sin(angles[0]+angles[1])*links[1] + Ex1
            Ey3 = math.cos(angles[0]+angles[1]+ angles[2]) * links[2] + Ey2
            Ex3 = math.sin(angles[0]+angles[1]+angles[2]) * links[2] + Ex2 

            joints = np.array([[0,Ex1,Ex2,Ex3],[0,Ey1,Ey2,Ey3]])
            #print("distance:", self.dist_to_goal)

            self.line1.set_data(joints[0],joints[1])
            self.goal_marker.set_data([self.goal[0]], [self.goal[1]])
            
            # Adjust axes limits dynamicall
            self.axes.set_xlim(-100, 100)
            self.axes.set_ylim(-100, 100)
            # Redraw the canvas
            self.fig.canvas.draw()


    def close(self):
        plt.close()

        
from stable_baselines3.common.env_util import DummyVecEnv     
from stable_baselines3.common.env_checker import check_env

   
                                                          
def make_env(rank):
    def _init():
        env = TwoD_Robot()  # Replace with your environment
        env = Monitor(env, filename=f"monitor_logs/env_{rank}") 
        return env 
    return _init


#env = DummyVecEnv([make_env(i) for i in range(2)])
#env = Monitor(env,filename="monitor_logs/")

env = TwoD_Robot()  # Replace with your environment
check_env(env)
env = Monitor(env, filename=f"monitor_logs/env_00") 

# Initialize the PPO model
model = PPO(policy = "MultiInputPolicy", env= env, n_steps=2048)                                 
# Custom training loop with renderin
n_episodes = 1200
max_steps = 180 # 00 is approximately the average steps for learing 
model.learn(total_timesteps=n_episodes*max_steps)
# Save the trained model
model.save("ppo_twod_robot", include_env=False)


import plot_results
plot_results.plot_monitor_data('monitor_logs/env_00.monitor.csv')

from test_model import test_model
test_model(model)