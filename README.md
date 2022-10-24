# Critic-Variational Autoencoder for Crafter
Fork of https://github.com/lcicek/Critic-VAE

Leverages the critic-model: [Critic](https://github.com/ndrwmlnk/critic-guided-segmentation-of-rewarding-objects-in-first-person-views)

Train a critic CNN and a VAE on crafter images.

# Installation and Usage

If you just want to train the models with different parameters you can use the google colab implemetations:

[Critic training on colab](https://colab.research.google.com/drive/14-KIMmQElpW2zbtTQOU2RlhOzl-rUfzM?usp=sharing) \
[VAE training on colab](https://colab.research.google.com/drive/1YoAEnPFhnybgOPynUT_ljcY_Hoh-ilkb?usp=sharing)

## Installation for Local training
You will need **python>=3.9.12** to run this. Just install the dependencies with 

```
python -m pip install -r requirements.txt
```

## Train Critic

After you installed it you can start the critic training with 

```
python crafter_extension_train_critic.py
```

## Train VAE

To train the VAE use

```
python vae.py -train-crafter -crafter-windowsize 20 -crafter-dataset-size 50000 -crafter-epochs 400
```
You can find more detailed info in my [Project report](Critic_guided_VAE_for_Crafter.pdf).
