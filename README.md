# YAPPO
## Yet Another implementation of Proximal Policy Optimisation

This class is based on ppo1 from the openai baselines but removes a bit of bloat and a few unnecessarily complicated algorithmic designs.

The model is an MLP with two hidden layers of 64 nodes so it doesn't take much computing to reach a respectable level of performance. I ran the training algorithm for about 10 mins in a single process after which the half cheetah achieved a decent looking gait.

## Installation

The main two dependencies are Roboschool and OpenAI baselines for which the installation instructions are [here](https://github.com/openai/roboschool#installation) and [here](https://github.com/openai/baselines#installation) respectively. 

## To train use:
`python main.py --env=$YOUR_ENV_NAME --ntimesteps=$PROBABLY_ABOUT_TWENTY_MILL`

## To watch the results use:
`python main.py --env=$YOUR_ENV_NAME --ntimesteps=$PROBABLY_ABOUT_TWO_THOU --train=False`
