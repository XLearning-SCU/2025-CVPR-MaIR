import torch
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from tqdm import tqdm

@MODEL_REGISTRY.register()
class MaIRPlusModel(SRModel):
    """MaIR model for image restoration."""
    def one_img_test(self, img):
        _, C, h, w = img.size()
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g_ema(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                return _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.net_g.train()
                _, _, h, w = _img.size()
                _img = _img[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
                return _img
    # test by partitioning

    def gather(self, imgs):
        for i in range(len(imgs), 0, -1):
            if i > 4:
                imgs[i-1] = imgs[i-1].clone().transpose(2,3)
            if (i-1) %4 > 1:
                imgs[i-1] = TF.hflip(imgs[i-1])
            if ((i-1) % 4) % 2 == 1:
                imgs[i-1] = TF.vflip(imgs[i-1])
        imgs = torch.cat(imgs, dim=0)
        imgs = torch.mean(imgs, dim=0, keepdim=True)
        return imgs
    
    def augment(self, img):
        imgs = [0] * 9
        for i in range(1,9):
            if i == 1:
                imgs[i] = img
            elif i == 2:
                imgs[i] = TF.vflip(img)
            elif i >2 and i <=4 :
                imgs[i] = TF.hflip(imgs[i-2])
            elif i > 4:
                imgs[i] = imgs[i-4].transpose(2,3)
        return imgs[1:]

    def test(self):
        lqs = self.augment(self.lq)
        output = []
        for i in tqdm(range(len(lqs))):
            output.append(self.one_img_test(lqs[i]))
        self.output = self.gather(output)
