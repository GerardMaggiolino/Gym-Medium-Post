import gym
import torch
from agent import TRPOAgent
import simple_driving
import time
import csv
'''
    
                
'''

def main():
    
    tParamBatchSize = [3000, 5000, 8000]
    tParamHiddenLayer = [32]
    tParamInitNoiseStd = [0.05, 0.1, 0.15]
    tParamNoiseChange = ["inputWeight", "outputWeight", "bothWeights"]

    n=1
    with open('../Data/Noisy Overnight/Param Recording/Parameters.csv', 'w', newline='') as modelCSV:
        r = csv.writer(modelCSV, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        r.writerow(["Model num", "Batch Size", "Hidden Layer Size","Noise std", "Noise Location"])
        for batchSize in tParamBatchSize:
            for hiddenLayer in tParamHiddenLayer:
                for initNoise in tParamInitNoiseStd:
                    for noiseChange in tParamNoiseChange:
                        if noiseChange == "inputWeight":
                             inputWeightNoise, outputWeightNoise = True, False
                        elif noiseChange == "outputWeight":
                            inputWeightNoise, outputWeightNoise = False, True
                        elif noiseChange == "bothWeights":
                            inputWeightNoise, outputWeightNoise = True, True                            
                        try:
                            nn = torch.nn.Sequential(torch.nn.Linear(8, hiddenLayer), torch.nn.Tanh(),
                                                    torch.nn.Linear(hiddenLayer, 2))
                        
                            #TODO: Add noise parameters into init (Check init for TRPO agents for which parameter is which)
                            agent = TRPOAgent(policy=nn, input_noise=False, output_noise=False, weight_one_noise=inputWeightNoise, weight_two_noise=outputWeightNoise, init_noise_std=initNoise, noise_anneal_epochs=3)

                            #agent.load_model("models/Model Reward=24.967.pth")
                            agent.train("SimpleDriving-v0", seed=0, batch_size=batchSize, iterations=3,
                                        max_episode_length=250, verbose=True, model_num=n)
                            agent.save_best_agent(f"../Data/Noisy Overnight/Models/Model #{n} ")
                            r.writerow([n, batchSize, hiddenLayer, initNoise, noiseChange])
                            modelCSV.flush()
                            n+=1
                    
                        except ZeroDivisionError:
                            r.writerow([n, 'Broke mate'])

    #agent.save_best_agent("models/")

    '''
    env = gym.make('SimpleDriving-v0')
    ob = env.reset()
    env.render()
    
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
