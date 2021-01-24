# Real_Simple_RL
Real Simple RL is a framework for conducting reinforcement learning experiments to examine the effect of errors in state abstraction on Q-learning, and to test whether the detachment algorithm, the central contribution of my thesis, ameliorates these errors. 

Currently the forms of state abstraction supported are the Q*, A*, and Pi* abstractions from the paper Towards a Unified Theory of State Abstraction (Li, Walsh, Littman). 

The environments supported are the 4 rooms environment, a customizable 2 rooms environment, and a simple MDP specifically constructed to demonstrated the performance of the detachment algorithm. 

To conduct an experiment, run run_experiment.py. Within this file, specify the environment (and parameters of the environment if using the TwoRoomsMDP) and parameters, such as which abstractions are run, specific errors or how many random errors to generate, how many agents in an ensemble, and so on. 

To run the visualizer, run either run_visualizer.py or run_two_rooms_visualizer.py (according to the MDP). 

I apologize if you somehow found this page as these instructions are quite incomplete, and visualizations of results will be forthcoming in my thesis. Please contact me at trevor.r.pearce@gmail.com if you have questions. 
