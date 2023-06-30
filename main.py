import gym
import torch
from agent import TRPOAgent
import simple_driving
import time
import csv


def main():

    tParamHiddenLayer = [48, 64, 96, 128]
    tParamBatchSize = [4000, 6000, 8000, 10000, 12000]
    tParamIterations = [50, 100, 150, 200]
    tParamEpisodeLength = [175, 250, 325]

    n=1
    with open('ModelTracking.csv', 'w', newline='') as modelCSV:

        r = csv.writer(modelCSV, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        r.writerow(["Models Name", "Hidden Layer", "Batch Size", "Iterations", "EpisodeLength"])

        for hiddenLayer in tParamHiddenLayer:
            for batchSize in tParamBatchSize:
                for iterations in tParamIterations:
                    for episodeLength in tParamEpisodeLength:
                        try:
                            nn = torch.nn.Sequential(torch.nn.Linear(8, hiddenLayer), torch.nn.Tanh(),
                                                    torch.nn.Linear(hiddenLayer, 2))
                            agent = TRPOAgent(policy=nn)

                            #agent.load_model("models/Model Reward=24.967.pth")
                            #agent.test_model("SimpleDriving-v0")
                            agent.train("SimpleDriving-v0", seed=0, batch_size=batchSize, iterations=iterations, 
                                        max_episode_length=episodeLength, verbose=True)
                            #agent.save_model("models/")
                            # Directories not existing and that fix it
                            agent.save_best_agent(f"../OverNightTraining/Model{n} ")
                            r.writerow([n, hiddenLayer, batchSize, iterations, episodeLength])
                            n+=1
                        except ZeroDivisionError:
                            r.writerow([n, 'A ZeroDivisionError has occured.'])

    #env = gym.make('SimpleDriving-v0')
    #ob = env.reset()
    
    '''
    while True:
        action = agent(ob)
        ob, _, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1/30)
    '''
    

if __name__ == '__main__':
    main()
