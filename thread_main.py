import gym
import torch
from agent import TRPOAgent
import simple_driving
import time
import csv
from queue import Queue
from multiprocessing import Process
from multiprocessing import Manager
'''
    
                
'''

def AgentWorker(queue):
    job =  queue.get()
    #try:
    #nn = torch.nn.Sequential(torch.nn.Linear(8, job["hidden_layers"]), torch.nn.Tanh(),
     #                       torch.nn.Linear(job["hidden_layers"], 2))
    nn = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.Tanh(),
                            torch.nn.Linear(64, 2))

    #TODO: Add noise parameters into init (Check init for TRPO agents for which parameter is which)
    agent = TRPOAgent(policy=nn, input_noise=False, output_noise=False, weight_one_noise=job["input_weight_noise"], weight_two_noise=job["output_weight_noise"], max_noise_std=job["init_noise"], max_epochs=100, anneal=job["anneal"])

    #agent.load_model("models/good base.pth")
    agent.train("SimpleDriving-v0", seed=0, batch_size=job["batch_size"], iterations=100,
                max_episode_length=250, verbose=True, model_num=0)
    #agent.save_best_agent(f"../Data/Noisy Overnight/Models/Model #{n} ")
        #r.writerow([n, batchSize, hiddenLayer, initNoise, noiseChange, anneal])
        #modelCSV.flush()
        #n+=1

    #except ZeroDivisionError:
        #r.writerow([n, 'Broke mate'])

def main():
    tParamBatchSize = [1500]
    tParamHiddenLayer = [64]
    tParamInitNoiseStd = [1.0, 1.5, 2.0, 4.0, 6.0]
    tParamNoiseChange = ["inputWeight","outputWeigh","bothWeights"]
    tParamAnnealNoise = [True, False]

    job_queue = Queue()

    with open('../Data/Noisy Overnight/Param Recording/Parameters.csv', 'w', newline='') as modelCSV:
        r = csv.writer(modelCSV, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        r.writerow(["Model num", "Batch Size", "Hidden Layer Size","Noise std", "Noise Location", "Annealment"])
        for batchSize in tParamBatchSize:
            for hiddenLayer in tParamHiddenLayer:
                for initNoise in tParamInitNoiseStd:
                    for noiseChange in tParamNoiseChange:
                        for anneal in tParamAnnealNoise:
                            if noiseChange == "inputWeight":
                                inputWeightNoise, outputWeightNoise = True, False
                            elif noiseChange == "outputWeight":
                                inputWeightNoise, outputWeightNoise = False, True
                            elif noiseChange == "bothWeights":
                                inputWeightNoise, outputWeightNoise = True, True
                            job_queue.put({"batch_size":batchSize, "hidden_layers":hiddenLayer, "init_noise":initNoise, "input_weight_noise":inputWeightNoise, "output_weight_noise":outputWeightNoise, "anneal":anneal})

    num_threads = 4
    for _ in range(num_threads):
        t = threading.Thread(target=AgentWorker, args=(job_queue, )) 
        t.daemon = True
        t.start()

    job_queue.join()

    print("Great job you have finished!")                     


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
