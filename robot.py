import gymnasium as gym 
from gymnasium import spaces 
import numpy as np 
import math 
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO   
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import pandas as pd


class TwoD_Robot(gym.Env): 
    
    def __init__(self):
        # stuff for training 
        self.max_episode_steps = 1000
        self.goal_reward = 1000
        self.current_step = 0
        self.current_episode =  0    
        self.episodic_rewards = 0 
        self.summary = []
        # length of robot link    
        self.L1 = 42.5 
        self.L2 = 39.22 
        self.L3 = 9.97
        
        #initial robot motor angles and endeffektor pose  
        self.theta = np.array([0.0,0.0,0.0])
        self.ee_pose = self.forward_kinematics(self.theta)
        self.dist_to_goal = 100
        self.angle_distance = 0
        
        # random goal initializeation
        self.goal = np.array([0,30,np.pi])
        
        # Action space: motorangles 
        self.action_space = spaces.Box(low= -1, high = 1, shape = (3,))
       
        # endeffektor pose, joint angles, goal, distance to goal space definition  
        self.observation_space = spaces.Dict(
            {
                "pose": spaces.Box(
                        low = np.array([-100,-100,-3*np.pi]),
                        high =np.array([100,100,3*np.pi]), 
                        dtype = float
                    ), 
                "motors": spaces.Box(
                        low = np.array([-np.pi, -np.pi, -np.pi]),
                        high =np.array([ np.pi, np.pi, np.pi]),
                        dtype = float
                    ),
                 "goal": spaces.Box(
                        low = np.array([-100,-100,-np.pi]),
                        high =np.array([100,100,np.pi]), 
                        dtype = float
                    ),
                "distance": spaces.Box(
                        low = np.array([0,0]),
                        high = np.array([200, 3*np.pi]),
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

        
        self.number_steps_episode = []
        self.episode_array = []
        self.fig2, self.ax2 = plt.subplots(1, 1)
        self.line2, = self.ax2.plot([],'.')
        self.ax2.set_title("steps per episodes")


    # function that returns ee_pose from given motorangles 
    def forward_kinematics(self,theta):
        """compute Endeffector Pose and returns x,y and theta (angles). zero pose is a vertical line"""
        x = (self.L1 * np.sin(theta[0]) +
            self.L2 * np.sin(theta[0] + theta[1]) +
            self.L3 * np.sin(theta[0] + theta[1] + theta[2]))
        y = (self.L1 * np.cos(theta[0]) +
            self.L2 * np.cos(theta[0] + theta[1]) +
            self.L3 * np.cos(theta[0] + theta[1] + theta[2]))
        theta_E = theta[0] + theta[1] + theta[2]
        return np.array([x,y,theta_E])
         
    # get observation / current state as dict with ee_pose, motorangles, goal, distance to goal 
    def _get_obs(self):
        self.ee_pose = self.forward_kinematics(self.theta)
        self.dist_to_goal = np.linalg.norm(self.ee_pose[:2] - self.goal[:2])
        self.angle_distance = np.abs(np.abs(self.ee_pose[2] - self.goal[2]))
        dist_total = np.array([self.dist_to_goal, self.angle_distance])
    
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
        r_d = 500/self.dist_to_goal 
        r_a = -0.1* self.angle_distance / np.pi *0
        
        # reward for keeping away from smashing the motors
        r_edge = - self.theta[0]**2 - self.theta[1]**2 - self.theta[2]**2
        # reward if done 
        r_goal= 0 
        if (self.dist_to_goal < 5):
            r_done = self.goal_reward - self.current_step/2
        
        r_terminated = 0     
        if (self.current_step == self.max_episode_steps-1):
            r_terminated = -500
   
        # print("r_edge:",r_edge)
        # print("r_d:", r_d)
        # print("r_done", r_done)
        reward = r_d + r_a + r_goal + r_edge + r_terminated         
        return reward        

    # set the robot to random state and defines new goal, returns new observation  and info(empty) 
    def reset(self, seed = None, options = None):
        #returns new random state and emptyinfo 
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        #self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.theta = np.array([0.0,0.0,0.0])#np.random.uniform(-np.pi, np.pi, size = 3)
        #set new goal 
        rad = np.sqrt(np.random.uniform(0,1))* 60
        angle = np.random.uniform(-np.pi,np.pi)
        x = rad * np.cos(angle)
        y = rad * np.sin(angle)
        theta_e = np.random.uniform(-np.pi,np.pi)         
        self.goal = np.array([x,y,theta_e])
        
        print("reset, goal: ", self.goal)

        #get observation and info 
    

        observation = self._get_obs()
        info = self._get_info()
        return observation, info 
      
    def check_done (self):
        done = False
        reason= ''
        self.episodic_rewards += self._get_reward()
        
        if (self.dist_to_goal < 5):
            done = True
            reason = 'goal reached'
        
        if (self.current_step>= self.max_episode_steps): 
            done = True
            self.current_step = -100 
            reason = 'max_episodes reached'
                   
        if done: 
            print("done epsiode:",self.current_episode)
            print(reason, "steps needed",self.current_step)
            self.number_steps_episode.append(self.current_step)
            self.episode_array.append(self.current_episode)
            self.current_step = 0 
            self.current_episode +=1 
            self.episodic_rewards = 0
            self.summary.append([self.current_episode,self.current_step,self.episodic_rewards])
            
        else: 
            self.current_step += 1    
            
        return done 
        
    #returns observation, reward, done = bool, truncated = False, info = empty  
    def step(self,action):
        # perform action and oberve 
        self.theta += action
        self.theta = np.clip(self.theta, -np.pi, np.pi)  # Keep angles in range
        self.ee_pose = self.forward_kinematics(self.theta)
    
        #observe
        observation = self._get_obs()
        reward = self._get_reward()
        done = self.check_done()
        truncated = False
        info = self._get_info()          
        return observation, reward, done, truncated, info 

    # redenring without returning anything 
    def render(self):

        goal = self.goal
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
        
        if done:
            self.line2.set_data(self.episode_array, self.number_steps_episode)
            self.ax2.relim()
            self.ax2.autoscale()
            self.fig2.canvas.draw()   
        
        plt.pause(0.001)
    
    def close(self):
        # Close the plot
        plt.close(self.fig)   

                                                                                                                              
env = TwoD_Robot()
# check if environmnet is working properly 
check_env(env)
env = Monitor(env,filename="monitor_logs/")

# Initialize the PPO model
model = PPO(policy = "MultiInputPolicy", env= env, ent_coef=0.01, verbose=1)                                 
# Custom training loop with rendering
total_timesteps = 10000


steps_per_render = 100  # Render every 100 steps
obs, info = env.reset()

for step in range(total_timesteps):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, done ,truncated, info = env.step(action)
    
    if done: 
        obs,info = env.reset()
    plt.ion()
    
    if (total_timesteps%steps_per_render == 0) :
        env.render()
    
data = env.summary     
df = pd.DataFrame(data,columns = "episodes, reward, steps")     

df.to_csv('data.csv', index=False)  # Set index=False to not write row numbers

