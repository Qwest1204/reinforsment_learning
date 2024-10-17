from model import *
from data import *
from load_config import *
from torchvision import transforms as T

configuration = get_config_for("VAE")

PATH_TO_WEIGHT = configuration['path_to_weight']
LATENT_DIM = configuration['latent_dim']
BATCH_SIZE = configuration['batch_size']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# First preprocessing of data
transform1 = T.Compose([T.Resize(64),
                        T.CenterCrop(64)])

# Data augmentation and converting to tensors
#random_transforms = [transforms.RandomRotation(degrees=10)]
transform2 = T.Compose([T.RandomHorizontalFlip(p=0.5),
                        #transforms.RandomApply(random_transforms, p=0.3), 
                        T.ToTensor(),
                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

def init_model(path_to_weight, latent_dim, batch_size):
    model = VAE(latent_dim, batch_size=batch_size).to(device)
    model = torch.load(path_to_weight)
    model.eval()
    
    return model

model = init_model(PATH_TO_WEIGHT, LATENT_DIM, BATCH_SIZE)

def prepare_image(img):
    img = transform2(transform1(img))
    return img

def jope(img):
    x = prepare_image(img)
    reconstructed, _, _ = model(x.to(device))
    reconstructed = reconstructed.view(-1, 3, 64, 64).detach().cpu().numpy().transpose(0, 2, 3, 1)
    return reconstructed