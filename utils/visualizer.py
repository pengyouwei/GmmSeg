import os
import numpy as np
import matplotlib.pyplot as plt
from config import Config


def create_visualization(image_show,        # [H, W]
                         label_show,        # [H, W]
                         mu_show,           # [K*C, H, W]
                         var_show,          # [K*C, H, W]
                         pi_show,           # [K, H, W]
                         d1_show,           # [K, H, W]
                         d0_show,           # [K, H, W]
                         pred_show,         # [H, W]
                         output_dir, 
                         epoch, 
                         logger=None):

    cfg = Config()
    os.makedirs(output_dir, exist_ok=True)

    c = 1
    if epoch % c != 0:
        return

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes[0, 0].imshow(image_show, cmap='gray')
    axes[1, 0].imshow(label_show, cmap='gray')
    axes[2, 0].imshow(pred_show, cmap='gray')
    axes[3, 0].imshow(label_show - pred_show, cmap='bwr', vmin=-1, vmax=1)
    axes[0, 0].set_title("Image")
    axes[1, 0].set_title("Label")
    axes[2, 0].set_title("Pred")
    axes[3, 0].set_title("Error Map")
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    axes[3, 0].axis('off')

    for i in range(1, 5):
        axes[0, i].imshow(d0_show[i-1], cmap='gray')
        axes[1, i].imshow(d1_show[i-1], cmap='gray')
        axes[2, i].imshow(d1_show[i-1], cmap='gray')
        axes[2, i].imshow(d0_show[i-1], cmap='jet', alpha=0.4)
        axes[3, i].imshow(pi_show[i-1], cmap='gray')
        axes[0, i].set_title(f"d0[{i-1}]")
        axes[1, i].set_title(f"d1[{i-1}]")
        axes[2, i].set_title(f"d1-d0[{i-1}]")
        axes[3, i].set_title(f"pi[{i-1}]")
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        axes[2, i].axis('off')
        axes[3, i].axis('off')

    fig.savefig(os.path.join(output_dir, f"vis_epoch_{epoch}.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
