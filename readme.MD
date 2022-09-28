# neuroevolution of self interpretable agent on procgen games
This project is the reimplementation of the attention agent from the paper "Neuroevolution of self interpretable agents" by Google.
This agent operates on the vectorized procgen gym environment and has been trained to play the starpilot game but can be trained to play all the games of the procgen environment.

# environment
The vectorized enviroment allows to play multiple games in parallel so the agent will not output a single action as output but one action for each environment and take as a reward the mean of the rewards minus a death loss that can be personalized to penalize deaths. So everything is batched starting from the input images that the agent sees till the output.

# attention and controllers
The attention mechanism is the one described by the paper but i tested different controllers to actually output the actions for example the mlp, the lstm controller and an attention controller developed by me.

# features for the network
The features that the agent sees from the patches with the most importance can be personalized and could be merged together having a little bit of everything. They can be personalized with :
the simple coordinates of the patch ,the color of the patch(mean of rgb channels) or they can be automatically detected with a feature extractor module that is a neural network ,also this could be changed.


# dependencies
install procgen cmaes and pytorch pickle wandb argparse
pip install procgen
pip install cma

the other commands can be found on the internet.

# training an agent
To train the agent with the cmaes algorithm refer to the parallelEvolution.py script.
There the game can be changed, the type of network too, the features too, death penality too and so on.
Setting startagain = True will make the training restart where it was left ,otherwise to start a new training there is the need to set this value to false.
The training can be done running once all the settings are selected using the command "python parallelEvolution.py"
You will see the best global performance as output and the mean value for each generation.


# visualize the agent in the environment
In order to see how the agent is performing refer to the testAgent.py script.
The parameters containing the weights can be selected . 
selecting parametes.pt the best global network is selected, with current.pt the current generation best network is selected, otherwise one could select from the paramters training or testing folder the parameters to use. In those paraeters in the title there is the encoding of all the settings used that have to be setted in the test agent script.

In the console while the agent is playing we can see where the agent is paying attention.
0 means it is not paying attention to the patch , 1 is the most important patch, 2 is the second most important patch and so on.

# running the agent pretrained
To run the networks trained with different settings the following comands can be executed:
"python testAgent.py"
that will run the network with features automatically extracted 
or add the args:
"--color","--extractedFeatures","--position","--attention","--deathPenalization"

the default controller is the lstm but the attentin one uses the attention controller.

# different version of the project 
in this folder there is the version with the environment not vectorized (its slower because it doesnt work in parallel with the gpu),
and there is the parallel evolution where the agent can be trained in parallel with multiprocessing.
with my gtx 1060 with 3gb of dram its not useful but maybe with 12 gb of dram it could be useful.
i added this just in case but some things have to be changed.
