# Ethical Extensions to a Federated Learning Framework for Single-Cell RNA Sequencing
 
This repository contains the implementation, experimental results, and the final report for the Ethics in Artificial Intelligence course project, academic year 2025/2026.

Authors: Foschi Virginia, Triulzi Giada

### Repository structure
 
```
AI_ETHICS_Proj_Foschi_Triulzi/
└── FL/
    ├── data_centralized/          
    ├── data_federated/            
    ├── exercise-hvg/              
    ├── exercise-scvi/            #directory containing the starting federated learning experiment extended with code for privacy and fairness experiments
    │   ├── app/
    │   │   ├── fairness/           
    │   │   ├── old/               
    │   │   ├── __init__.py
    │   │   ├── client_app.py     
    │   │   ├── custom_strategy.py 
    │   │   ├── server_app.py      
    │   │   └── task.py
    │   ├── centralized_training.ipynb
    │   ├── models_comparison.ipynb
    │   ├── preliminaries.py       # Downloads/prepares datasets required for the experiments
    │   └── pyproject.toml         # Project config: dependencies + experiment parameters           
    ├── fairness_analysis/     # Notebooks/scripts for the fairness experiments
    ├── fairness_results/      # Precomputed fairness experiment outputs (models, logs)
    ├── privacy_analysis/      # Notebooks/scripts for the privacy experiments
    ├── pp_results/            # Precomputed partial-participation experiment outputs
    ├── model_centralized/     
    ├── model_federated/       
    ├── preliminaries.py       # Downloads/prepares datasets required for the experiments        
    ├── fl-course-env.yaml     # Conda environment specification
    ├── fl-course-env-new.yaml # Updated Conda environment specification
    ├── loss_plot.png
    ├── .gitignore
    └── README.md
```


## 1. Environment and data preparation
 
1. Create a folder named `data_centralized` and download the dataset from
   the following link into it:
   https://drive.switch.ch/index.php/s/J8NLl6KBVWwQEEA
2. Create the required Conda environment from the supplied environment
   file (`fl-course-env.yaml`), which specifies all required Python
   packages and their dependencies:
```bash
   conda env create -f fl-course-env.yaml
   conda activate fl-course-env
```
 
3. Run the preliminaries script to download and prepare the datasets
   required for the subsequent experiments:
```bash
   python preliminaries.py
```
 
   If necessary, navigate to the correct project folder first

 
## 2. Experiment selection
 
All experiments implemented in `exercise-scvi` are selected through the
`strategy` field in `exercise-scvi/pyproject.toml`. This field accepts one
of three values:
 
```toml
strategy = "privacy"
strategy = "fairness"
strategy = "availability"
```
 
These correspond, respectively, to:
 
- the **privacy** experiments (Client Identification Attack, Property Inference Attack and Differential Privacy),
- the **client-fairness** experiments (comparison of aggregation
  strategies),
- the **partial client participation** experiments (client dropout
  regimes).
  
Additional fields in `exercise-scvi/pyproject.toml`, described in the
sections below, control the specific configuration of the selected
experiment.

## 3. Privacy experiments (`strategy = "privacy"`)
 
Before running these experiments, download the appropriate
`client_updates` folder from Google Drive:
 
- The **first** link contains the updates required to reproduce the
  Client Identification Attack and the **untargeted** Differential
  Privacy experiments with noise multipliers of `0.3` and `0.6`:
  https://drive.google.com/drive/folders/1mceP2WSYdzPZ1_d-4IFkVhOUaL7eYCZd?usp=share_link
- The **second** link provides the updates required for the experiments
  with a noise multiplier of `1` and for **targeted** Differential
  Privacy:
  https://drive.google.com/drive/folders/1Nh8gyXAbQMi1LuRLRVn_0xrcBqThbXET?usp=share_link

After downloading the desired `client_updates` folder, copy it inside the
directory named `FL`, replacing the existing folder if necessary. **Do
not** change the `client_updates` folder name.
 
Once the folder is in place, open the `attacks_and_mitigation.ipynb`
notebook, located in the `privacy_analysis` folder, and run the cells
sequentially to reproduce the corresponding experiments and inspect the
results.
 
Alternatively, the `client_updates` folder can be generated from scratch
by selecting `strategy = "privacy"` and rerunning the federated training.
 
To experiment with different Differential Privacy configurations, modify
`dp_noise_multiplier` and `dp_clip_norm` in
`exercise-scvi/pyproject.toml` according to the values you wish to test.
Set `dp_noise_multiplier = 0` if you do not want Differential Privacy to
be applied.
 
In the current version of the code, the privacy strategy is configured to
apply targeted Differential Privacy, reflecting the more favorable
privacy–utility trade-off observed in our experiments. To switch between
untargeted and targeted noise injection, comment/uncomment the call to
`add_dp_noise` or `add_dp_noise_targeted` respectively, inside the
`train()` function of `exercise-scvi/app/client_app.py`.
 
 
## 4. Fairness experiments (`strategy = "fairness"`)
 
When `strategy = "fairness"` is selected, the aggregation strategy is
specified through the `fairness_strategy_name` parameter. The available
options are:
 
```toml
fairness_strategy_name = "qffl"
fairness_strategy_name = "adaptive_qffl"
fairness_strategy_name = "uniform"
fairness_strategy_name = "fedavg_ploss"
```
 
For q-FFL, the parameter `fairness_q` specifies the initial value of $q$.
The parameters `fairness_q_alpha`, `fairness_q_min`, and
`fairness_q_max` control the behaviour of the adaptive q-FFL variant by
defining the step size, lower bound, and upper bound of $q$,
respectively.
 
The fairness experiments can therefore be reproduced by selecting the
desired aggregation strategy and adjusting the corresponding parameters
in `pyproject.toml` before launching the experiment.
 
The trained models and corresponding outputs for the fairness experiments
are provided in the `fairness_results` directory. The fairness analysis
notebooks can therefore be executed directly using the provided results,
without requiring the federated training process to be run again.
These notebooks allow the user to investigate the results for the
selected aggregation strategy and reproduce the client-level metrics,
fairness measures, and visualizations presented in the report.

 
## 5. Partial client participation experiments (`strategy = "availability"`)
 
Partial client participation is selected using `strategy = "availability"`.
The participation regime is controlled by `participation_regime`, which
can take the values `"full"`, `"random"`, or `"size_correlated"`.
 
- For the **random** participation regime, `availability_p` specifies
  the probability that a client is available in each communication
  round.
- For the **size-correlated** regime, clients with fewer than
  `availability_size_threshold` cells are considered small clients and
  are assigned probability `availability_p_small`, while larger clients
  are assigned `availability_p_large`. The configuration used in the
  reported experiments was:
```toml
  availability_size_threshold = 850
  availability_p_small = 0.5
  availability_p_large = 0.9
```
 
The aggregation strategy used during the availability experiments is
controlled by `availability_q`. Setting `availability_q = 0.0`
corresponds to FedAvg-equivalent weighting, whereas larger values
introduce the q-FFL weighting mechanism.
 
The random participation process is controlled by `availability_seed`,
ensuring that the dropout sequence can be reproduced. Finally,
`availability_bootstrap_rounds` specifies the number of initial rounds
during which all clients participate before the selected dropout regime
is activated.
 
As with the fairness experiments, the trained models and results are
already provided — in this case, in the `pp_results` directory. The
corresponding analysis can therefore be reproduced directly using the
provided notebook (`partial_partecipation_results.ipynb`), without
rerunning the federated training experiments.
 
 
