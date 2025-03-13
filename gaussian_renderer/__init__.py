#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.dual_quaternion import quaternion_mul
import seaborn as sns
import numpy as np


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz= None, d_rot= None, scaling_modifier=1.0, random_bg_color=False, scale_const=None, mask=None, vis_mask=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    bg = bg_color if not random_bg_color else torch.rand_like(bg_color)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    xyz = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    sh_features = pc.get_features

    means3D = xyz + d_xyz if d_xyz is not None else xyz
    means2D = screenspace_points 
    if scale_const is not None:
        opacity = torch.ones_like(pc.get_opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(d_rot, scaling_modifier)
    else:
        rotations = quaternion_mul(d_rot, rotations) if d_rot is not None else rotations

    if mask != None:
        shs = None
        pallete = torch.from_numpy(np.array(sns.color_palette("hls", mask.max() + 1))).float().to(pc.get_xyz.device)
        colors_precomp = pallete[mask]
    else:
        shs = sh_features
        colors_precomp = None

    if scale_const is not None:
        scales = scale_const * torch.ones_like(scales)

    # Rasterize visible Gaussians to image.
    if vis_mask is not None:
        means3D = means3D[vis_mask]
        means2D = means2D[vis_mask]
        shs = shs[vis_mask] if shs is not None else None
        colors_precomp = colors_precomp[vis_mask] if colors_precomp is not None else None
        opacity = opacity[vis_mask]
        scales = scales[vis_mask]
        rotations = rotations[vis_mask]
        cov3D_precomp = cov3D_precomp[vis_mask] if cov3D_precomp is not None else None

    rendered_image, radii, depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "alpha": alpha,
            "bg_color": bg}