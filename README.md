# Curation Modeling

This repository contains the implementation of the Cura system.

## Setting up

Check out the code from our repository using 

```bash
git clone https://github.com/Azure-Vision/Curation-Modeling.git
git checkout Trans
```

Place `.env.development` in the root directory.

Install the dependencies using 

```bash
conda env create -f environment.yml
```

Then activate the environment with

```bash
conda activate cr2
```

## Train on Reddit data

Run the following script

```bash
python train.py CONFIG_PATH
```

The configurations used in the paper “Cura: Curation at Social Media Scale” can be found at `configs/subreddit_minority_no_peer_new.yml`, and the configurations for the online experiment that includes more subreddits can be found at `configs/subreddit_minority_no_peer_more_subs.yml`.

## Run prediction on Reddit data

Evaluate the prediction accuracy and confidence of the curation model under different conditions: run `test_model.ipynb`.

Evaluate the change in prediction accuracy when the curation model receives more peer votes: run `sim_new_votes.ipynb`.

Perform curation on selected subreddit given selected curators: run `curation.ipynb`.

Luanch the interface for administrators to select curators and perform curation using

```bash
streamlit run curation_interface.py
```

## Integrate to Curio

Collect and preprocess posting and user data from Curio app: run `process_CURIO_data.ipynb`.

Finetune the pretrained curation model on Curio data using

```bash
cd trained_models; mkdir finetune_CURIO_full_data; mkdir deploy_CURIO_full_data; cp subreddit_minority_no_peer_new/latest.pt finetune_CURIO_full_data/latest.pt; cd ..; python train.py configs/finetune_CURIO_full_data.yml; cp trained_models/finetune_CURIO_full_data/best.pt trained_models/deploy_CURIO_full_data/best.pt
```

Launch the curation model backend for Curio with

```bash
uvicorn curation_backend:app --port 5000
```
