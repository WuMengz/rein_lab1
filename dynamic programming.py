import gym # openAi gym
import numpy as np 
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def gamedemo(env):
    for t in range(100):
        env.render() # 渲染画面
        a = env.action_space.sample() # 随机采样动作
        observation, reward, done, info = env.step(a) # 环境执行动作，获得转移后的状态、奖励以及环境是否终止的指示
        print(observation, reward, done, info)
        if done:
            break

def policy_evaluation(policy, env, gamma=1.0, theta=0.00001):
    """
    实现策略评估算法，给定策略与环境模型，计算该策略对应的价值函数。

    参数：
      policy：维度为[S, A]的矩阵，用于表示策略。
      env：gym环境，其env.P表示了环境的转移概率。
        env.P[s][a]为一个列表，其每个元素为一个表示转移概率以及奖励函数的元组(prob, next_state, reward, done)
        env.observation_space.n表示环境的状态数。
        env.action_space.n表示环境的动作数。
      gamma：折扣因子。
      theta：用于判定评估是否停止的阈值。
    
    返回值：长度为env.observation_space.n的数组，用于表示各状态的价值。
    """
    
    nS = env.observation_space.n
    nA = env.action_space.n

    # 初始化价值函数
    V = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            v_new = 0
            for a in range(nA):
              for prob, next_state, reward, done in env.P[s][a]:
                v_new+=policy[s][a] * prob * (reward + gamma*V[next_state])
            
            delta = max(delta, np.abs(V[s]-v_new))
            V[s] = v_new
        # 误差小于阈值时终止计算
        if delta < theta:
            break
      
    return np.array(V)

def policy_iteration(env, policy_eval_fn=policy_evaluation, gamma=1.0):
    """
    实现策略提升算法，迭代地评估并提升策略，直到收敛至最优策略。

    参数：
      env：gym环境。
      policy_eval_fn：策略评估函数。
      gamma：折扣因子。

    返回值：
      (policy, V)
      policy为最优策略，由维度为[S, A]的矩阵进行表示。
      V为最优策略对应的价值函数。
    """

    nS = env.observation_space.n
    nA = env.action_space.n

    def one_step_lookahead(state, V):
        """
        对于给定状态，计算各个动作对应的价值。
        
        参数：
            state：给定的状态 (int)。
            V：状态价值，长度为env.observation_space.n的数组。
        
        返回值：
            每个动作对应的期望价值，长度为env.action_space.n的数组。
        """
        A = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + gamma * V[next_state])

        return A

    # 初始化为随机策略
    policy = np.ones([nS, nA]) / nA
    
    num_iterations = 0

    while True:
        num_iterations += 1
        
        V = policy_eval_fn(policy, env, gamma)
        policy_stable = True
        
        for s in range(nS):
            old_action = np.argmax(policy[s])

            q_values = one_step_lookahead(s, V)
            new_action = np.argmax(q_values)

            if old_action != new_action:
                policy_stable = False
                        
            policy[s] = np.zeros([nA])
            policy[s][new_action] = 1

        if policy_stable:
            print(num_iterations)
            return policy, V
  
def value_iteration(env, theta=0.00001, gamma=1.0):
    """
    实现价值迭代算法。
    
    参数：
      env：gym环境，其env.P表示了环境的转移概率。
        env.P[s][a]为一个列表，其每个元素为一个表示转移概率以及奖励函数的元组(prob, next_state, reward, done)
        env.observation_space.n表示环境的状态数。
        env.action_space.n表示环境的动作数。
      gamma：折扣因子。
      theta：用于判定评估是否停止的阈值。
        
    返回值：
      (policy, V)
      policy为最优策略，由维度为[S, A]的矩阵进行表示。
      V为最优策略对应的价值函数。       
    """

    nS = env.observation_space.n
    nA = env.action_space.n
    
    def one_step_lookahead(state, V):
        """
        对于给定状态，计算各个动作对应的价值。
        
        参数：
            state：给定的状态 (int)。
            V：状态价值，长度为env.observation_space.n的数组。
        
        返回值：
            每个动作对应的期望价值，长度为env.action_space.n的数组。
        """
        A = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + gamma * V[next_state])

        return A
    
    V = np.zeros(nS)
    
    num_iterations = 0
    
    while True:
        num_iterations += 1
        delta = 0
        
        for s in range(nS):
            q_values = one_step_lookahead(s, V)
            new_value = np.max(q_values)
            
            delta = max(delta, np.abs(new_value - V[s]))
            V[s] = new_value
        
        if delta < theta:
            break
    
    policy = np.zeros([nS, nA])
    for s in range(nS): 
        q_values = one_step_lookahead(s,V)
        
        new_action = np.argmax(q_values)
        policy[s][new_action] = 1
    
    print(num_iterations * 16)    
    return policy, V, num_iterations * nS

def value_iteration_priority(env, theta=0.00001, gamma = 1.0):
    
    nS = env.observation_space.n
    nA = env.action_space.n
    bellman_errors = np.zeros(nS)
    
    def one_step_lookahead(state, V):
        """
        对于给定状态，计算各个动作对应的价值。
        
        参数：
            state：给定的状态 (int)。
            V：状态价值，长度为env.observation_space.n的数组。
        
        返回值：
            每个动作对应的期望价值，长度为env.action_space.n的数组。
        """
        A = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + gamma * V[next_state])

        return A

    def update_bellman_errors(V):
        for s in range(nS):
            q_values = one_step_lookahead(s, V)
            new_val = np.max(q_values)
            bellman_errors[s] = abs(V[s] - new_val)
 
    V = np.zeros(nS)
    update_bellman_errors(V)
    
    num_iterations = 0

    while True:
        num_iterations += 1
        max_error = np.max(bellman_errors)
        if max_error < theta:
            break
        s = np.argmax(bellman_errors)
        q_values = one_step_lookahead(s, V)
        V[s] = np.max(q_values)
        update_bellman_errors(V)

    policy = np.zeros([nS, nA])
    for s in range(nS): 
        q_values = one_step_lookahead(s, V)
        
        new_action = np.argmax(q_values)
        policy[s][new_action] = 1
    
    print(num_iterations)
    return policy, V, num_iterations


def same_policy(policy1, policy2):
    nS = len(policy1)
    nA = len(policy1[0])
    for s in range(nS):
        for a in range(nA):
            if policy1[s][a] != policy2[s][a]:
                return False
    return True
  

env = gym.make("FrozenLake-v1")
env.reset()

# gamedemo(env)
# exit(0)

env.reset()
policyPI, valuePI = policy_iteration(env, gamma=0.95)
# print(policyPI)
# print(valuePI)

env.reset()
policyVI, valueVI, _ = value_iteration(env, theta = 1e-10, gamma=0.5)
# print(policyVI)
# print(valueVI)

env.reset()
policyVIP, valueVIP, _ = value_iteration_priority(env, theta = 1e-10, gamma=0.5)

if same_policy(policyVI, policyVIP):
    print("策略迭代算法与价值迭代算法的最终策略一致。")
else:
    print("策略迭代算法与价值迭代算法的最终策略不一致。")

def plot_iterations(env):
    thetas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    gammas = [0.33, 0.4, 0.45, 0.5, 0.6, 0.66, 0.7, 0.75, 0.78, 0.8, 0.83, 0.85, 0.87, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1]
    speedup = []
    for theta in thetas:
        step1 = value_iteration(env, theta=theta, gamma=1)[2]
        step2 = value_iteration_priority(env, theta=theta, gamma=1)[2]
        speedup.append(step1 / step2)
    plt.plot(["1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9", "1e-10"], speedup)
    plt.xlabel("theta")
    plt.ylabel("speedup")
    plt.show()
    speedup = []
    for gamma in gammas:
        step1 = value_iteration(env, theta=1e-8, gamma=gamma)[2]
        step2 = value_iteration_priority(env, theta=1e-8, gamma=gamma)[2]
        speedup.append(step1 / step2)
    plt.plot(gammas, speedup)
    plt.xlabel("gamma")
    plt.ylabel("speedup")
    plt.show()

plot_iterations(env)