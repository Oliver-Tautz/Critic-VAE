# The VAE has too little resolution ...


# 1 Use bigger latent dim

... kinda works but it still wont reconstruct small details :(

# 2  Use cleaner dataset

... Just gets more green and lava(red) will be reconstructed as water(blue) 

# 3 use more powerful decoder (Conv2dTranspose instead of upsample)

much better resolution, but no detail is remembered?! Maybe still bad dataset?

# 4 use more powerful encoder (conv2d instead of maxpool)

much better resolution, but no detail is remembered?! Maybe still bad dataset?

# 5 Take out batch normalization 

... not much of a difference

# 6 Try cleaner dataset, or different model?!
