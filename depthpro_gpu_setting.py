import torch
import depth_pro
def configure_gpu():

    # Load model and preprocessing transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    model = model.to(device)
    return model, transform, device