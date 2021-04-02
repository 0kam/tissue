from src.models import GMVAE
from src.utils import set_patches 

set_patches("data/labels2", "data/images_full", "data/patches/images_full", (3,3), batch_size=100)

gmvae = GMVAE("/home/okamoto/gmvae-segmentation/data/semisup/tateyama3D_3x3_full_2", "data/labels2", z_dim=4, batch_size=50, \
    drop_out_rate=0.0, lr=1e-3, device="cuda", num_workers=20)

gmvae.train(10, "test_full_0402")
gmvae.draw("data/images_snow_small", "results/test_snow_small.png", (3,3), 5000)
gmvae.draw_legend("results/legend.png")
gmvae.draw_teacher("results/teacher.png", (1684, 1123))

