import gym
import torch
from agent import TRPOAgent
import simple_driving
import time
import csv

def main():
    
    tParamBatchSize = [5000]
    tParamHiddenLayer = [64]
    tParamInitNoiseStd = [0.5, 1.0, 1.5, 2.0]
    tParamNoiseChange = ["ActionNoise", "ObservationNoise"]
    tParamAnnealNoise = [True, False]
    n=1
    with open('../Data/Noisy Overnight/Param Recording/Parameters.csv', 'w', newline='') as modelCSV:
        r = csv.writer(modelCSV, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        r.writerow(["Model num", "Batch Size", "Hidden Layer Size","Noise std", "Noise Location", "Annealment"])
        for anneal in tParamAnnealNoise:
            for batchSize in tParamBatchSize:
                for hiddenLayer in tParamHiddenLayer:
                    for initNoise in tParamInitNoiseStd:
                        for noiseChange in tParamNoiseChange: 

                            #Noise Truth Table   
                            if noiseChange == "inputWeight":
                                inputWeightNoise, outputWeightNoise = True, False
                            elif noiseChange == "outputWeight":
                                inputWeightNoise, outputWeightNoise = False, True
                            elif noiseChange == "bothWeights":
                                inputWeightNoise, outputWeightNoise = True, True       

                            if noiseChange == "ActionNoise":
                                actionNoise, ObservationNoise = True, False
                            elif noiseChange == "ObservationNoise":
                                actionNoise, ObservationNoise = False, True
                            try:
                                nn = torch.nn.Sequential(torch.nn.Linear(8, hiddenLayer), torch.nn.Tanh(),
                                                        torch.nn.Linear(hiddenLayer, 2))
                            
                                #TODO: Add noise parameters into init (Check init for TRPO agents for which parameter is which)
                                agent = TRPOAgent(policy=nn, input_noise=actionNoise, output_noise=ObservationNoise, weight_one_noise=False, weight_two_noise=False, max_noise_std=initNoise, max_epochs=199, anneal=anneal)

                                #agent.load_model("models/good base.pth")
                                agent.train("SimpleDriving-v0", seed=0, batch_size=batchSize, iterations=200,
                                            max_episode_length=250, verbose=True, model_num=n)
                                agent.save_best_agent(f"../Data/Noisy Overnight/Models/Model #{n} ")
                                r.writerow([n, batchSize, hiddenLayer, initNoise, noiseChange, anneal])
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
