from VAE.data import *
from VAE.model import *
from torchvision import transforms as T
from PIL import Image

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

transform3 = T.Compose([
                        T.ToPILImage(),
                        T.Resize((64, 64)),
                        #T.RandomRotation(degrees=90),
                        T.ToTensor(),])

def init_model(path_to_weight, latent_dim, batch_size):
    """
        Initializes and loads a pre-trained VAE model.

        Args:
            path_to_weight (str): Path to the file containing the saved model weights.
            latent_dim (int): Dimensionality of the latent space.
            batch_size (int): Batch size used during training.

        Returns:
            VAE: Initialized and loaded VAE model.
    """
    model = VAE(latent_dim, batch_size=batch_size).to(device)
    model.load_state_dict(torch.load(path_to_weight, weights_only=True))
    model.eval()
    
    return model

def prepare_image(img):
    """
        Prepares an image for use in a machine learning model.

        Args:
            img (numpy.ndarray): The input image as a NumPy array.

        Returns:
            torch.Tensor: The transformed image as a PyTorch tensor.
    """
    PIL_image = Image.fromarray(img)
    img = transform3(PIL_image)
    return img

def compute_image(img, model):
    """
        Reconstructs an image using a pre-trained VAE model.

        Args:
            img (numpy.ndarray): The input image to be reconstructed.
            model: Initialized and loaded VAE model.

        Returns:
            numpy.ndarray: The reconstructed image as a NumPy array.
    """
    x = prepare_image(img)
    reconstructed, _, _ = model(x[None, :, :, :].to(DEVICE).to(device))
    reconstructed = reconstructed.view(-1, 3, 64, 64).detach().cpu().numpy().transpose(0, 2, 3, 1)
    return reconstructed


def compute_Z(img, model):
    """
        Computes the latent representation (Z) of an image using a VAE model.

        Args:
            img (numpy.ndarray): The input image.
            model (VAE): The pre-trained VAE model.

        Returns:
            torch.Tensor: The computed latent representation (Z) of the image.
    """
    x = prepare_image(img)
    z = model.encode(x[None, :, :, :].to(DEVICE))
    return z

def reconstruct_Z_by_t(Z, t, model):
    """
        Reconstructs an image from a latent representation at a specific time step.

        Args:
            Z: A numpy array containing latent representations for all time steps.
            t: The time step for which to reconstruct the image.
            model: The VAE model

        Returns:
            A numpy array representing the reconstructed image.
    """
    img = model.decoder(torch.from_numpy(Z[t]).to(DEVICE)).detach().cpu().numpy().transpose(0, 2, 3, 1)
    return img

def reconstruct_Z(Z, model):
    """
        Reconstructs images from latent representations for all time steps.

        Args:
            Z: A numpy array containing latent representations for all time steps.
            model: The diffusion model used for reconstruction.

        Returns:
            A list of numpy arrays representing the reconstructed images for each time step.
    """
    image = []
    total_z_parametr = Z.shape[0]

    for t in range(total_z_parametr):
        image.append(reconstruct_Z_by_t(Z, t, model))
        
    return image