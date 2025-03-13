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

import os
import sys
import tqdm
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func, vis_depth
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.log_utils import prepare_output_and_logger, training_report
from pytorch_lightning import seed_everything
from utils.metrics import *
from pytorch3d.loss import chamfer_distance
from utils.depth_loss import DepthLoss


class Trainer:
    def __init__(self, args, dataset, opt, pipe, saving_iterations):
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.saving_iterations = saving_iterations

        self.tb_writer = prepare_output_and_logger(args)
        
        self.gaussians = GaussianModel(dataset.sh_degree)
        self.deform = DeformModel(self.dataset)
        print('Init GaussianModel and DeformModel.')
        self.scene = Scene(dataset, self.gaussians, load_iteration=-1)

        if self.args.canonical_init == 'cgs':
            p = args.source_path.replace('data/', 'outputs/')
            coarse_name = self.args.coarse_name
            self.xyzs = self.gaussians.load_ply_cano(f'{p}/{coarse_name}/point_cloud/iteration_10000/point_cloud.ply')
            print('Init canonical gaussians from coarse gaussian.')
        else:
            print('Init canonical gaussians randomly.')

        self.init_deform()
        self.gaussians.training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1 if self.scene.loaded_iter is None else self.scene.loaded_iter

        self.viewpoint_stacks = [self.scene.getTrainCameras_start(), self.scene.getTrainCameras_end()]
        
        self.ema_loss_for_log = 0.0
        self.best_iteration = 15000
        self.best_joint_error = 1e10
        self.joint_metrics = []

        self.progress_bar = tqdm.tqdm(range(self.iteration-1, opt.iterations), desc="Training progress")
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

        self.cd_loss_weight = args.cd_loss_weight
        self.metric_depth_loss_weight = args.metric_depth_loss_weight
        self.mono_depth_loss_weight = args.mono_depth_loss_weight

        self.depth_loss = DepthLoss()
    
    def init_deform(self,):
        if self.args.center_init == 'cgs':
            p = args.source_path.replace('data/', 'outputs/')
            coarse_name = self.args.coarse_name
            center, scale = self.deform.deform.seg_model.init_from_file(f'{p}/{coarse_name}/point_cloud/iteration_10000/center_info.npy')
            # import torch.nn as nn
            # joints = torch.randn_like(self.deform.deform.joints) * 1e-5
            # for i in range(len(joints)):
            #     if self.deform.deform.joint_types[i+1] == 'r':
            #         joints[i, 4:7] += center[0].clone()
            # joints[:, 0] = 1
            # self.deform.deform.joints = nn.Parameter(joints)
            print('Init center from coarse gaussian.')
        elif self.args.center_init == 'pcd':
            center, scale = self.deform.deform.seg_model.init_from_file(f'{self.args.source_path}/center_info.npy')
            print('Init center from pcd.')
        else:
            print('Init center randomly.')
        self.deform.load_weights(self.dataset.model_path, iteration=-1)
        self.deform.train_setting(self.opt)
    
    def train(self, iters=5000):
        for i in tqdm.trange(iters):
            self.train_step()

    def train_step(self):
        self.iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()
            
        state = randint(0, 1)
        id = randint(0, len(self.viewpoint_stacks[state]) - 1)
        viewpoint_cam = self.viewpoint_stacks[state][id]
        
        # Render
        random_bg = (not self.dataset.white_background and self.opt.random_bg_color) and viewpoint_cam.gt_alpha_mask is not None
        bg = self.background if not random_bg else torch.rand_like(self.background).cuda()
        d_values = self.deform.deform.one_transform(self.gaussians, state, is_training=True)
        d_xyz, d_rot = d_values['d_xyz'], d_values['d_rotation']
        render_pkg_re = render(viewpoint_cam, self.gaussians, self.pipe, bg, d_xyz=d_xyz, d_rot=d_rot)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        if random_bg:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * bg[:, None, None]
        elif self.dataset.white_background and viewpoint_cam.gt_alpha_mask is not None:
            gt_image = gt_alpha_mask * gt_image + (1 - gt_alpha_mask) * self.background[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # point loss
        if self.cd_loss_weight > 0 and self.iteration > self.args.cd_min_steps and self.iteration < self.args.cd_max_steps and 'p' not in self.dataset.joint_types:
            xt = self.gaussians.get_xyz.detach() + d_xyz
            cd, _ = chamfer_distance(xt[None], self.xyzs[state][None], single_directional=True)
            cd_loss = self.cd_loss_weight * cd
            loss = loss + cd_loss   
            self.tb_writer.add_scalar('train/cd_loss', cd_loss.item(), self.iteration)


        # depth loss
        depth_loss = torch.tensor([0.])
        if self.metric_depth_loss_weight > 0:
            depth = render_pkg_re['depth']
            gt_depth = viewpoint_cam.depth.cuda()
            invalid_mask = (gt_depth < 0.1) & (gt_alpha_mask > 0.5)
            valid_mask = ~invalid_mask
            n_valid_pixel = valid_mask.sum()
            if n_valid_pixel > 100:
                depth_loss = (torch.log(1 + torch.abs(depth - gt_depth)) * valid_mask).sum() / n_valid_pixel
                loss = loss + depth_loss * self.metric_depth_loss_weight

        mono_depth_loss = torch.tensor([0.])
        if self.mono_depth_loss_weight > 0:
            depth = render_pkg_re['depth']
            mono_depth = viewpoint_cam.mono_depth.cuda()
            # mono_depth_loss = depth_rank_loss(depth, mono_depth, gt_alpha_mask)
            mono_depth_loss = self.depth_loss(depth, mono_depth[None], gt_alpha_mask)
            loss = loss + mono_depth_loss * self.mono_depth_loss_weight

        if self.iteration > 3000:
            loss = loss + self.deform.reg_loss
        
        loss.backward()
        self.iter_end.record()

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{6}f}"})
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            if self.iteration % 1000 == 0:
                try:
                    joint_types = self.deform.deform.joint_types[1:]
                    pred_joint_list = self.deform.deform.get_joint_param(joint_types)
                    gt_info_list = read_gt(os.path.expanduser(f'{args.source_path}/gt/trans.json'))
                    self.joint_metrics, real_perm = eval_axis_and_state_all(pred_joint_list, joint_types, gt_info_list)
                except:
                    print('No ground truth info for joint evaluation.')
            # # Log and save
            training_report(self.tb_writer, self.iteration, Ll1, depth_loss, mono_depth_loss, loss, 
                            self.iter_start.elapsed_time(self.iter_end), self.scene, self.joint_metrics)
            if self.iteration % 100 == 0 and self.iteration > 15000:
                cur_joint_error = sum([sum(m) for m in self.joint_metrics]) if len(self.joint_metrics) > 0 else 1e5
                if cur_joint_error < self.best_joint_error or (self.iteration == self.args.iterations and self.best_iteration <= 15000):
                    self.best_iteration = self.iteration
                    self.best_joint_error = cur_joint_error
                
            if self.iteration in self.saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration)
                self.deform.save_weights(self.args.model_path, self.iteration)
            if self.iteration == self.best_iteration:
                print("\n[ITER {}] Saving Gaussians".format(self.iteration))
                self.scene.save(self.iteration, is_best=True)
                self.deform.save_weights(self.args.model_path, self.iteration, is_best=True)
            
            # Keep track of max radii in image-space for pruning
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            # Densification
            if self.iteration < self.opt.densify_until_iter:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    threshold = 0.01 if self.iteration > 3000 else 0.005
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, threshold, self.scene.cameras_extent, size_threshold)
                
                if self.iteration % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            self.gaussians.optimizer.step()
            self.gaussians.update_learning_rate(self.iteration)
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
            self.deform.update_learning_rate(self.iteration)
            
            self.deform.update(max(0, self.iteration))

        self.iteration += 1

    def visualize(self, image, gt_image, gt_depth, depth):
        torchvision.utils.save_image(image.detach(), "img.png")
        torchvision.utils.save_image(gt_image, "img_gt.png")
        torchvision.utils.save_image(vis_depth(gt_depth), "gt.png")
        torchvision.utils.save_image(vis_depth(depth.detach()), "pred.png")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, 20_000, 40_000, 60_000, 80_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    args.joint_types = json.load(open(f'./arguments/joint_types_{args.center_init}.json', 'r'))[args.dataset][args.subset][args.scene_name]
    args.num_slots = len(args.joint_types.split(','))
    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args), saving_iterations=args.save_iterations)
    trainer.train(args.iterations)
    print("\nTraining complete.")
