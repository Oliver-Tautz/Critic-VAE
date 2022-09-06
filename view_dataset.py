from crafter_extension_utils import load_crafter_pictures
from tqdm import tqdm


from PIL import Image

path = '/home/olli/gits/Critic-VAE/dataset/'

pics = load_crafter_pictures(path,windowsize=20)

for i,pic in enumerate(tqdm(pics)):
    pic = Image.fromarray(pic)
    pic.save(f'dataset/pics/{i}.png')