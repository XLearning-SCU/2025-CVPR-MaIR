import time
import torch 
from einops import rearrange
import numpy as np

def test_crop_by_horz(inp, scan_len):
    # Flip the return way
    split_inp = rearrange(inp, "h (d1 w) -> d1 h w ", w=scan_len)
    for i in range(1, len(split_inp), 2):
        split_inp[i, :] = split_inp[i, :].flip(dims=[-2])
    inp = rearrange(split_inp, " d1 h w -> (d1 h) w ")
    # print(inp)

    inp_window = rearrange(inp, "(d1 h) (d2 w) -> (d2 d1) h w ", h=2, w=scan_len)

    inp_window[:,-1] = inp_window[:,-1].flip(dims=[-1])
    inp_flatten = inp_window.reshape(1, -1)
    print(inp_flatten)
    print(inp_flatten.shape)

def chw_2d(h, w):
    return torch.arange(1, (h*w+1)).reshape(h, w)

def chw_3d(c, h, w):
    return torch.arange(1, (c*h*w+1)).reshape(c, h, w)

def chw_4d(b, c, h, w, random=False):
    if random:
        return torch.randn(b*c*h*w).reshape(b, c, h, w)
    else:
        return torch.arange(1, (b*c*h*w+1)).reshape(b, c, h, w)

def create_idx(b, c, h, w):
    # return torch.arange(1, (b*c*h*w+1)).reshape(b, c, h, w)
    return torch.arange(b*c*h*w).reshape(b, c, h, w)

def test_2d_horz(inp_h, inp_w):
    scan_len = 2
    # inp_h, inp_w = 4, 4
    # inp  = torch.randn((4,4))
    inp = torch.tensor([[ 1,  2,  3,  4],
                        [ 5,  6,  7,  8],
                        [ 9,  10, 11, 12],
                        [ 13, 14, 15, 16]])
    inp = chw_2d(inp_h, inp_w)
    print(inp)
    test_crop_by_horz(inp, scan_len)

def sscan_einops(inp, scan_len):
    B, C, H, W = inp.shape
    # Flip the return way
    split_inp = rearrange(inp, "b c h (d1 w) -> d1 b c h w ", w=scan_len)
    for i in range(1, len(split_inp), 2):
        split_inp[i, :] = split_inp[i, :].flip(dims=[-2])
    reverse_inp = rearrange(split_inp, " d1 b c h w -> b c (d1 h) w ")
    # print(inp)

    inp_window = rearrange(reverse_inp, "b c (d1 h) (d2 w) -> (b c d2 d1) h w ", h=2, w=scan_len)

    inp_window[:,-1] = inp_window[:,-1].flip(dims=[-1])
    inp_flatten = inp_window.reshape(B, C, 1, -1)
    # print(inp_flatten)
    # print(inp_flatten.shape)

    return inp_flatten

def sscan(inp, scan_len, shift_len=0):
    B, C, H, W = inp.shape
    # Flip the return way
    # 将返回的时候的列，上下翻转
    if shift_len == 0:
        for i in range(1, (W // scan_len)+1, 2):
            # for j in range(scan_len):
            inp[:, :, :, i*scan_len:(i+1)*scan_len] = inp[:, :, :, i*scan_len:(i+1)*scan_len].flip(dims=[-2])
    else:
        for i in range(0, ((W-shift_len) // scan_len) +1, 2):
            inp[:, :, :,(shift_len+i*scan_len):(shift_len+(i+1)*scan_len)] = inp[:, :, :, (shift_len+i*scan_len):(shift_len+(i+1)*scan_len)].flip(dims=[-2])


    # 将当前return way的sub-line翻转
    # inp_window = rearrange(inp, "b c (d1 h) (d2 w) -> (b c d2 d1) h w ", h=2, w=scan_len)
    if shift_len == 0:
        for hi in range((H // 2)):
            for wi in range(W // scan_len):
                inp[:, :, 2*hi+1, wi*scan_len:(wi+1)*scan_len] = inp[:, :, 2*hi+1, wi*scan_len:(wi+1)*scan_len].flip(dims=[-1])
    else:
        for hi in range((H // 2)):
            inp[:, :, 2*hi+1, 0:shift_len] = inp[:, :, 2*hi+1, 0:shift_len].flip(dims=[-1])

            for wi in range((W-shift_len) // scan_len):
                start_ = shift_len + wi*scan_len
                end_ = shift_len + (wi+1)*scan_len
                inp[:, :, 2*hi+1, start_:end_] = inp[:, :, 2*hi+1, start_:end_].flip(dims=[-1])

        
    if (W-shift_len) % scan_len:
        # inp_last = inp[:,:,:,-(W % scan_len):].reshape(B, C, -1)
        inp_last = inp[:,:,:,-((W-shift_len) % scan_len):]
        inp_last[:,:, 1::2, :] =  inp_last[:,:, 1::2, :].flip(dims=[-1]) # 取偶数位，奇数位是::2
        inp_last = inp_last.reshape(B, C, -1)

        inp_rest = inp[:,:,:,:-((W-shift_len) % scan_len)]
    else:
        inp_rest = inp

    if shift_len==0:
        inp_window = rearrange(inp_rest, "b c h (d2 w) -> (b c d2) h w ", w=scan_len)
    else:
        inp_first = inp_rest[:,:,:,:shift_len].reshape(B, C, -1)

        inp_middle = inp_rest[:,:,:, shift_len:]
        inp_window = rearrange(inp_middle, "b c h (d2 w) -> (b c d2) h w ", w=scan_len)

    # inp_window[:,-1] = inp_window[:,-1].flip(dims=[-1])
    inp_flatten = inp_window.reshape(B, C, -1)
    # inp_window[:,-1] = inp_window[:,-1].flip(dims=[-1])
    # inp_flatten = inp.reshape(B, C, 1, -1)
    # print(inp_flatten)
    # print(inp_flatten.shape)
    if shift_len != 0:
        inp_flatten = torch.concat((inp_first, inp_flatten), dim=-1)

    if (W-shift_len) % scan_len:
        inp_flatten = torch.concat((inp_flatten, inp_last), dim=-1)
        # print(inp_last.shape)
    return inp_flatten


# def sscan_4d(inp, scan_len, ues_einops=True, fix_ending=True):
def sscan_4d(inp, scan_len, shift_len=0, fix_ending=True, use_einops=False):
    B, C, H, W = inp.shape
    L = H * W
    if fix_ending:
        inp_reverse = torch.flip(inp, dims=[-1,-2])
        inp_cat = torch.concat((inp, inp_reverse), dim=1)
        inp_cat_t = inp_cat.transpose(-1, -2).contiguous()

        if use_einops:
            line1 = sscan_einops(inp_cat, scan_len)
            line2 = sscan_einops(inp_cat_t, scan_len)
        else:
            line1 = sscan(inp_cat, scan_len, shift_len)
            line2 = sscan(inp_cat_t, scan_len, shift_len)

        xs = torch.stack([line1.reshape(B, 2, -1, L), line2.reshape(B, 2, -1, L)], dim=1).reshape(B, 4, -1, L)
    else:
        inp_t = inp.transpose(-1, -2).contiguous()
        if use_einops:
            line1 = sscan_einops(inp, scan_len)
            line2 = sscan_einops(inp_t, scan_len)
        else:
            line1 = sscan(inp, scan_len, shift_len)
            line2 = sscan(inp_t, scan_len, shift_len)

        x_hwwh = torch.stack([line1.reshape(B, -1, L), line2.reshape(B, -1, L)], dim=1).reshape(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) 
    # print(xs)
    return xs

def inverse_ids_generate(origin_ids, K=4):
    '''
        Input: origin_ids: (B, K, C, L)
        Output: (B, K, C, L)
        Note: C is set to 1 for speeding up.
    '''
    inverse_ids = torch.argsort(origin_ids, dim=-1)
    return inverse_ids

def mair_ids_generate(inp_shape, scan_len=4, K=4):
    inp_b, inp_c, inp_h, inp_w = inp_shape

    # inp_idx = create_idx(1, inp_c, inp_h, inp_w)
    inp_idx = create_idx(1, 1, inp_h, inp_w)

    xs_scan_ids = sscan_4d(inp_idx, scan_len)[0]

    xs_inverse_ids = inverse_ids_generate(xs_scan_ids, K=K)

    return xs_scan_ids, xs_inverse_ids


def mair_shift_ids_generate(inp_shape, scan_len=4, shift_len=0, K=4):
    inp_b, inp_c, inp_h, inp_w = inp_shape

    # create_idx函数运行时间：0.0050699710845947266 秒
    # start_time = time.time()
    inp_idx = create_idx(1, 1, inp_h, inp_w)
    # print(f"create_idx函数运行时间：{time.time() - start_time} 秒")

    # start_time = time.time()
    xs_scan_ids = sscan_4d(inp_idx, scan_len, shift_len=shift_len)[0]
    # print(f"sscan_4d函数运行时间：{time.time() - start_time} 秒")

    # xs_scan_ids函数运行时间：0.05201005935668945 秒
    # start_time = time.time()
    xs_scan_ids = xs_scan_ids.repeat(inp_b, 1, 1, 1)
    # print(f"xs_scan_ids函数运行时间：{time.time() - start_time} 秒")

    # start_time = time.time()
    xs_inverse_ids = inverse_ids_generate(xs_scan_ids, K=K)
    # print(f"inverse_ids_generate函数运行时间：{time.time() - start_time} 秒")

    return xs_scan_ids, xs_inverse_ids


def mair_ids_scan(inp, xs_scan_ids, bkdl=False, K=4):
    '''
        inp: B, C, H, W
        xs_scan_ids: K, 1, L
    '''
    B, C, H, W = inp.shape
    L = H * W
    xs_scan_ids = xs_scan_ids.reshape(K, L)

    y1 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[0])
    y2 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[1])
    y3 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[2])
    y4 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[3])

    if bkdl:
        inp_flatten = torch.cat((y1, y2, y3, y4), dim=1)
    else:
        inp_flatten = torch.cat((y1, y2, y3, y4), dim=1).reshape(B, 4, -1)
    return inp_flatten

def mair_ids_inverse(inp, xs_scan_ids, shape=None):
    '''
        inp: (B, K, -1, L)
        xs_scan_ids: (1, K, 1, L)
    '''
    B, K, _, L = inp.shape
    xs_scan_ids = xs_scan_ids.reshape(K, L)
    if not shape:
        y1 = torch.index_select(inp[:, 0, :], -1, xs_scan_ids[0]).reshape(B, -1, L)
        y2 = torch.index_select(inp[:, 1, :], -1, xs_scan_ids[1]).reshape(B, -1, L)
        y3 = torch.index_select(inp[:, 2, :], -1, xs_scan_ids[2]).reshape(B, -1, L)
        y4 = torch.index_select(inp[:, 3, :], -1, xs_scan_ids[3]).reshape(B, -1, L)
    else:
        B, C, H, W = shape
        y1 = torch.index_select(inp[:, 0, :], -1, xs_scan_ids[0]).reshape(B, -1, H, W)
        y2 = torch.index_select(inp[:, 1, :], -1, xs_scan_ids[1]).reshape(B, -1, H, W)
        y3 = torch.index_select(inp[:, 2, :], -1, xs_scan_ids[2]).reshape(B, -1, H, W)
        y4 = torch.index_select(inp[:, 3, :], -1, xs_scan_ids[3]).reshape(B, -1, H, W)
    return torch.cat((y1,y2,y3,y4), dim=1)


def test_time():
    scan_len = 4
    shift_len = 2
    inp_b, inp_c, inp_h, inp_w = 2, 3, 3, 4 
    inp = chw_4d(1, 1, inp_h, inp_w, random=False)
    inp_rgb = chw_4d(inp_b, inp_c, inp_h, inp_w, random=False)
    print("inp:", inp_rgb)

    # Original
    xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp.shape, scan_len=scan_len, K=4)
    xs = mair_ids_scan(inp_rgb, xs_scan_ids, bkdl=True)
    inp_flatten = mair_ids_inverse(xs, xs_inverse_ids, shape=(inp_b, inp_c, inp_h, inp_w))

    inp_flatten = inp_flatten.chunk(4, dim=1)
    print("recovered input:")
    for i in range(len(inp_flatten)):
        print("inp_flatten:", i)
        print(inp_flatten[i])
    print("end")


if __name__ == '__main__':
    # torch.set_default_device(1)
    start_time = time.time()
    result = test_time()
    end_time = time.time()

    print(f"函数运行时间：{end_time - start_time} 秒")
