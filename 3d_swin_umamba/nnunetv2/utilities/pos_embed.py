import torch
import torch.nn.functional as F
import numpy as np

def get_1d_sincos_pos_embed_from_grid(pos, embed_dim):

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    # original_shape = pos.shape
    # pos = pos.reshape(-1)  # (M,)
    # pos = pos.squeeze(0) 
    out = np.einsum('mln,d->mlnd', pos, omega) 
    # out = out.reshape( embed_dim // 2, *original_shape)  # (D, H, W, D/2)
    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    # emb = np.concatenate([emb_sin, emb_cos], axis=0)  # (M, D)
    emb = np.zeros((pos.shape[0], pos.shape[1], pos.shape[2], embed_dim), dtype=np.float32)
    emb[:, :, :, 0::2] = emb_sin
    emb[:, :, :, 1::2] = emb_cos
    return emb


def get_3d_sincos_pos_embed( d, h, w, dim):

    grid_d = np.arange(d, dtype=np.float32)
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_d, grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(3, d, h, w)

    pos_embed = get_3d_sincos_pos_embed_from_grid(grid, dim)
    return pos_embed
    
def get_3d_sincos_pos_embed_from_grid(grid, dim):
    assert dim % 3 == 0, "Dimension must be divisible by 3"
    
    d_embed = get_1d_sincos_pos_embed_from_grid(grid[0], dim//3)
    h_embed = get_1d_sincos_pos_embed_from_grid(grid[1], dim//3)
    w_embed = get_1d_sincos_pos_embed_from_grid(grid[2], dim//3)

    return np.concatenate([d_embed, h_embed, w_embed], axis=-1)

if __name__ == "__main__":
    d, h, w, dim = 32, 32, 32, 96
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dtype = torch.float32

    pos_embed = get_3d_sincos_pos_embed(d, h, w, dim)
    print(pos_embed.shape)  # Should be (1, 32, 32, 32, 96)