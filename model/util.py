import ipdb
import torch
import imageio
import torch.nn as nn
import torch.nn.functional as F


class Subspace(nn.Module):
    def __init__(self, ch_in, k=16, use_SVD=True, use_PCA=False):

        super(Subspace, self).__init__()
        self.k = k
        self.Block = SubspaceBlock(ch_in, self.k)
        self.use_SVD = use_SVD
        self.use_PCA = use_PCA

    def forward(self, x):

        sub = self.Block(x)
        x = subspace_project(x, sub)

        return x


class SubspaceBlock(nn.Module):
    def __init__(self, inplanes, planes):

        super(SubspaceBlock, self).__init__()

        self.relu = nn.LeakyReLU(inplace=True)

        self.conv0 = conv(
            inplanes, planes, kernel_size=1, stride=1, dilation=1, isReLU=False
        )
        self.bn0 = nn.BatchNorm2d(planes)
        self.conv1 = conv(
            planes, planes, kernel_size=1, stride=1, dilation=1, isReLU=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv(
            planes, planes, kernel_size=1, stride=1, dilation=1, isReLU=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = self.relu(self.bn0(self.conv0(x)))

        out = self.conv1(residual)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        return out


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ShareFeature_new(nn.Module):
    def __init__(self, inc=1, ouc=1, hic=16, kernel=7):
        super().__init__()
        pad = int(kernel / 2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(inc, hic, kernel, 1, pad, bias=False),
            nn.BatchNorm2d(hic),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hic, ouc, kernel, 1, pad, bias=False),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ShareFeature_new_v2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ShareFeature_ori(nn.Module):
    def __init__(self, inc=1, ouc=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inc, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, ouc, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ouc),
            nn.ReLU(inplace=True),
        )
        # 有点浅，海水移动这种feature要多几层conv

    def forward(self, x):
        x = self.layers(x)
        return x


# ========================================================================================================================
# Some functions that are not used here, which are designed for homography computation, may be helpful in some other works.


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):

    if isReLU:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((kernel_size - 1) * dilation) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((kernel_size - 1) * dilation) // 2,
                bias=True,
            )
        )


def initialize_msra(modules):

    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

        elif isinstance(layer, nn.LeakyReLU):
            pass

        elif isinstance(layer, nn.Sequential):
            pass


def subspace_project(input, vectors):

    b_, c_, h_, w_ = input.shape
    basis_vector_num = vectors.shape[1]
    V_t = vectors.view(b_, basis_vector_num, h_ * w_)
    V_t = V_t / (1e-6 + V_t.abs().sum(dim=2, keepdim=True))
    V = V_t.permute(0, 2, 1)
    mat = torch.bmm(V_t, V)
    mat_inv = torch.inverse(mat)
    project_mat = torch.bmm(mat_inv, V_t)
    input_ = input.view(b_, c_, h_ * w_)
    project_feature = torch.bmm(project_mat, input_.permute(0, 2, 1))
    output = torch.bmm(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)

    return output


def gen_basis(h, w, qr=True, scale=True):
    # multi device
    N = 8
    d_range = 10
    a_range = 0.2
    t_range = 0.5
    p_range = 0.001
    grid = get_grid(1, h, w).permute(0, 2, 3, 1)
    flows = grid[:, :, :, :2] * 0

    for i in range(N):
        # initial H matrix
        dx, dy, ax, ay, px, py, tx, ty = 0, 0, 0, 0, 0, 0, 1, 1

        if i == 0:
            dx = d_range
        if i == 1:
            dy = d_range
        if i == 2:
            ax = a_range
        if i == 3:
            ay = a_range
        if i == 4:
            tx = t_range
        if i == 5:
            ty = t_range
        if i == 6:
            px = p_range
            fm = 1  # grid[:, :, :, 0] * px + 1
            grid_ = grid.clone().float()
            grid_[:, :, :, 0] = grid_[:, :, :, 0] ** 2 * px / fm
            grid_[:, :, :, 1] = grid[:, :, :, 0] * grid[:, :, :, 1] * px / fm
            flow = grid_[:, :, :, :2]

        elif i == 7:
            py = p_range
            fm = 1  # grid[:, :, :, 1] * py + 1
            grid_ = grid.clone().float()
            grid_[:, :, :, 1] = grid_[:, :, :, 1] ** 2 * py / fm
            grid_[:, :, :, 0] = grid[:, :, :, 0] * grid[:, :, :, 1] * py / fm
            flow = grid_[:, :, :, :2]
        else:
            H_mat = torch.tensor([[tx, ax, dx], [ay, ty, dy], [px, py, 1]]).float()
            H_mat = H_mat.cuda() if torch.cuda.is_available() else H_mat
            # warp grids
            H_mat = H_mat.unsqueeze(0).repeat(h * w, 1, 1).unsqueeze(0)
            grid_ = (
                grid.reshape(-1, 3).unsqueeze(0).unsqueeze(3).float()
            )  # shape: 3, h*w
            grid_warp = torch.matmul(H_mat, grid_)
            grid_warp = grid_warp.squeeze().reshape(h, w, 3).unsqueeze(0)

            flow = grid[:, :, :, :2] - grid_warp[:, :, :, :2] / grid_warp[:, :, :, 2:]
        flows = torch.cat((flows, flow), 0)

    flows = flows[1:, ...]
    if qr:
        flows_ = flows.reshape(8, -1).permute(
            1, 0
        )  # N, h, w, c --> N, h*w*c --> h*w*c, N
        flows_q, _ = torch.qr(flows_)
        flows_q = flows_q.permute(1, 0).reshape(8, h, w, 2)
        flows = flows_q

    if scale:
        max_value = flows.abs().reshape(8, -1).max(1)[0].reshape(8, 1, 1, 1)
        flows = flows / max_value

    return flows.permute(0, 3, 1, 2)  # 8,h,w,2-->8,2,h,w


def get_grid(batch_size, H, W, start=0):

    if torch.cuda.is_available():
        xx = torch.arange(0, W).cuda()
        yy = torch.arange(0, H).cuda()
    else:
        xx = torch.arange(0, W)
        yy = torch.arange(0, H)
    xx = xx.view(1, -1).repeat(H, 1)
    yy = yy.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    ones = (
        torch.ones_like(xx).cuda() if torch.cuda.is_available() else torch.ones_like(xx)
    )
    grid = torch.cat((xx, yy, ones), 1).float()

    grid[:, :2, :, :] = grid[:, :2, :, :] + start  # add the coordinate of left top
    return grid


def get_src_p(batch_size, patch_size_h, patch_size_w, divides, axis_t=False):

    small_gap_sz = [patch_size_h // divides, patch_size_w // divides]
    mesh_num = divides + 1
    if torch.cuda.is_available():
        xx = torch.arange(0, mesh_num).cuda()
        yy = torch.arange(0, mesh_num).cuda()
    else:
        xx = torch.arange(0, mesh_num)
        yy = torch.arange(0, mesh_num)

    xx = xx.view(1, -1).repeat(mesh_num, 1)
    yy = yy.view(-1, 1).repeat(1, mesh_num)
    xx = xx.view(1, 1, mesh_num, mesh_num) * small_gap_sz[1]
    yy = yy.view(1, 1, mesh_num, mesh_num) * small_gap_sz[0]
    xx[:, :, :, -1] = xx[:, :, :, -1] - 1
    yy[:, :, -1, :] = yy[:, :, -1, :] - 1
    if axis_t:
        ones = (
            torch.ones_like(xx).cuda()
            if torch.cuda.is_available()
            else torch.ones_like(xx)
        )
        src_p = torch.cat((xx, yy, ones), 1).repeat(batch_size, 1, 1, 1).float()
    else:
        src_p = torch.cat((xx, yy), 1).repeat(batch_size, 1, 1, 1).float()

    return src_p


def chunk_2D(img, h_num, w_num, h_dim=2, w_dim=3):

    bs, c, h, w = img.shape
    img = img.chunk(h_num, h_dim)
    img = torch.cat(img, dim=w_dim)
    img = img.chunk(h_num * w_num, w_dim)
    return torch.cat(img, dim=1).reshape(bs, c, h_num, w_num, h // h_num, w // w_num)


def get_point_pairs(src_p, divide):  # src_p: shape=(bs, 2, h, w)

    bs = src_p.shape[0]
    src_p = src_p.repeat_interleave(2, axis=2).repeat_interleave(2, axis=3)
    src_p = src_p[:, :, 1:-1, 1:-1]
    src_p = chunk_2D(src_p, divide, divide).reshape(bs, -1, 2, 2, 2)
    src_p = src_p.permute(0, 1, 3, 4, 2).reshape(bs, divide * divide, 4, 2)
    return src_p


def dlt_homo(src_pt, dst_pt, method="Axb"):
    """
    :param src_pt: shape=(batch, num, 2)
    :param dst_pt:
    :param method: Axb (Full Rank Decomposition, inv_SVD) = 4 piar points
                Ax0 (SVD) >= 4 pair points, 4,6,8
    :return: Homography, shape=(batch, 3, 3)
    """
    assert method in ["Ax0", "Axb"]
    assert src_pt.shape[1] >= 4
    assert dst_pt.shape[1] >= 4
    if method == 'Axb':
        assert src_pt.shape[1] == 4
        assert dst_pt.shape[1] == 4
    batch_size, nums_pt = src_pt.shape[0], src_pt.shape[1]
    xy1 = torch.cat((src_pt, src_pt.new_ones(batch_size, nums_pt, 1)), dim=-1)
    xyu = torch.cat((xy1, xy1.new_zeros((batch_size, nums_pt, 3))), dim=-1)
    xyd = torch.cat((xy1.new_zeros((batch_size, nums_pt, 3)), xy1), dim=-1)
    M1 = torch.cat((xyu, xyd), dim=-1).view(batch_size, -1, 6)
    M2 = torch.matmul(dst_pt.view(-1, 2, 1), src_pt.view(-1, 1, 2)).view(
        batch_size, -1, 2
    )
    M3 = dst_pt.view(batch_size, -1, 1)

    if method == "Ax0":
        A = torch.cat((M1, -M2, -M3), dim=-1)
        U, S, V = torch.svd(A)
        V = V.transpose(-2, -1).conj()
        H = V[:, -1].view(batch_size, 3, 3)
        H = H * (1 / H[:, -1, -1].view(batch_size, 1, 1))
    elif method == "Axb":
        A = torch.cat((M1, -M2), dim=-1)
        B = M3
        A_inv = torch.inverse(A)
        H = torch.cat(
            (
                torch.matmul(A_inv, B).view(-1, 8),
                src_pt.new_ones((batch_size, 1)),
            ),
            1,
        ).view(batch_size, 3, 3)

    return H


def DLT_solve(src_p, off_set):

    bs, _, divide = src_p.shape[:3]
    divide = divide - 1

    src_ps = get_point_pairs(src_p, divide)
    off_sets = get_point_pairs(off_set, divide)

    bs, n, h, w = src_ps.shape
    N = bs * n

    src_ps = src_ps.reshape(N, h, w)
    off_sets = off_sets.reshape(N, h, w)

    dst_p = src_ps + off_sets

    ones = (
        torch.ones(N, 4, 1).cuda() if torch.cuda.is_available() else torch.ones(N, 4, 1)
    )
    xy1 = torch.cat((src_ps, ones), axis=2)
    zeros = (
        torch.zeros_like(xy1).cuda()
        if torch.cuda.is_available()
        else torch.zeros_like(xy1)
    )
    xyu, xyd = torch.cat((xy1, zeros), axis=2), torch.cat((zeros, xy1), axis=2)

    M1 = torch.cat((xyu, xyd), axis=2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), axis=2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.inverse(A)

    h8 = torch.matmul(Ainv, b).reshape(N, 8)

    H = torch.cat((h8, ones[:, 0, :]), axis=1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)
    return H


def get_flow(
    H_mat_mul, patch_indices, patch_size_h, patch_size_w, divide, point_use=False
):

    batch_size = H_mat_mul.shape[0]
    small_gap_sz = [patch_size_h // divide, patch_size_w // divide]

    small = 1e-7

    H_mat_pool = H_mat_mul.reshape(batch_size, divide, divide, 3, 3)  # .transpose(2,1)
    H_mat_pool = H_mat_pool.repeat_interleave(
        small_gap_sz[0], axis=1
    ).repeat_interleave(small_gap_sz[1], axis=2)

    if point_use and H_mat_pool.shape[2] != patch_indices.shape[2]:
        H_mat_pool = H_mat_pool.permute(0, 3, 4, 1, 2)
        H_mat_pool = F.pad(H_mat_pool, pad=(0, 1, 0, 1, 0, 0), mode="replicate")
        H_mat_pool = H_mat_pool.permute(0, 3, 4, 1, 2)

    pred_I2_index_warp = patch_indices.permute(0, 2, 3, 1).unsqueeze(
        4
    )  # 把bs, 2, h, w 换为 bs, h, w, 2

    pred_I2_index_warp = (
        torch.matmul(H_mat_pool, pred_I2_index_warp).squeeze(-1).permute(0, 3, 1, 2)
    )
    T_t = pred_I2_index_warp[:, 2:3, ...]
    smallers = 1e-6 * (1.0 - torch.ge(torch.abs(T_t), small).float())
    T_t = T_t + smallers
    v1 = pred_I2_index_warp[:, 0:1, ...]
    v2 = pred_I2_index_warp[:, 1:2, ...]
    v1 = v1 / T_t
    v2 = v2 / T_t
    pred_I2_index_warp = torch.cat((v1, v2), 1)
    vgrid = patch_indices[:, :2, ...]

    flow = pred_I2_index_warp - vgrid
    return flow, vgrid


def transformer(I, vgrid, train=True):
    # I: Img, shape: batch_size, 1, full_h, full_w
    # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
    # outsize: (patch_h, patch_w)

    def _interpolate(im, x, y, out_size):
        # x: x_grid_flat
        # y: y_grid_flat
        num_batch, num_channels, height, width = im.size()

        out_height, out_width = out_size[0], out_size[1]
        zero = 0
        max_y = height - 1
        max_x = width - 1

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        dim1 = width * height
        dim2 = width

        if torch.cuda.is_available():
            base = torch.arange(0, num_batch).int().cuda()
        else:
            base = torch.arange(0, num_batch).int()

        base = base * dim1
        base = base.repeat_interleave(out_height * out_width, axis=0)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im = im.permute(0, 2, 3, 1)
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _transform(I, vgrid):

        C_img = I.shape[1]
        B, C, H, W = vgrid.size()

        x_s_flat = vgrid[:, 0, ...].reshape([-1])
        y_s_flat = vgrid[:, 1, ...].reshape([-1])
        out_size = vgrid.shape[2:]
        input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size)

        output = input_transformed.reshape([B, H, W, C_img])
        return output

    output = _transform(I, vgrid)
    if train:
        output = output.permute(0, 3, 1, 2)
    return output


def get_warp(img, mesh, divide, grid_h=None, grid_w=None, start=0):

    batch_size, _, patch_size_h, patch_size_w = img.shape

    if grid_h is None:
        grid_h, grid_w = patch_size_h, patch_size_w

    src_p = get_src_p(batch_size, grid_h, grid_w, divide)
    patch_indices = get_grid(batch_size, grid_h, grid_w, 0)

    H_mat_mul = DLT_solve(src_p, mesh)

    flow, vgrid = get_flow(H_mat_mul, patch_indices, grid_h, grid_w, divide)

    grid_warp = get_grid(batch_size, grid_h, grid_w, start)[:, :2, :, :] - flow
    img_warp = transformer(img, grid_warp)
    return img_warp, flow


def get_warp_flow(img, flow, start=0):

    batch_size, _, patch_size_h, patch_size_w = flow.shape
    grid_warp = (
        get_grid(batch_size, patch_size_h, patch_size_w, start)[:, :2, :, :] - flow
    )
    img_warp = transformer(img, grid_warp)
    return img_warp


def upsample2d_flow_as(inputs, target_as, mode="bilinear", if_rate=False):

    _, _, h, w = target_as.size()
    if if_rate:
        _, _, h_, w_ = inputs.size()
        inputs[:, 0, :, :] *= w / w_
        inputs[:, 1, :, :] *= h / h_
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    return res


def warp_image_from_H(homo, img, batch_size, h_patch, w_patch):
    grid = get_grid(batch_size, h_patch, w_patch)
    flow, vgrid = get_flow(homo, grid, h_patch, w_patch, 1)
    grids = vgrid + flow
    # img_warp = transformer(img, grids)
    grids = grids.permute(0, 2, 3, 1)
    grids[..., 0] = grids[..., 0] / img.shape[3] * 2 - 1
    grids[..., 1] = grids[..., 1] / img.shape[2] * 2 - 1
    img_warp = F.grid_sample(img, grids, mode='bilinear')
    return img_warp


def warp_image_from_H_start(homo, img, batch_size, h_patch, w_patch, start=0):
    grid = get_grid(batch_size, h_patch, w_patch, start)
    flow, vgrid = get_flow(homo, grid, h_patch, w_patch, 1)
    grids = vgrid + flow
    img_warp = transformer(img, grids)
    return img_warp


def create_gif(image_list, gif_name, duration=0.5):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
