#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
import random
import time
import torch
import numpy as np

import deepsdf.deep_sdf as deep_sdf
import deepsdf.deep_sdf.workspace as ws

def reconstruct(
    decoder,
    num_iterations,
    latent_size,
    test_sdf,
    stat,
    clamp_dist,
    num_samples=30000,
    lr=5e-4,
    l2reg=False,
):
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
    ):
        # print(num_iterations, decreased_by, adjust_lr_every)
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        # latent = torch.normal(stat[0].detach(), stat[1].detach(), out=latent_size).cuda()
        latent = torch.empty(1, latent_size).normal_(mean=stat[0].detach(),std=stat[1].detach()).cuda()

    latent.requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    # loss_l1 = torch.nn.L1Loss(reduction="sum")
    loss_l2 = torch.nn.MSELoss(reduction="sum")

    for e in range(num_iterations):

        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            test_sdf, num_samples
        ).cuda()

        sdf_data = sdf_data.float()

        xyz = sdf_data[:, 0:3]
        
        # centering
        # center = torch.mean(xyz, axis=0)
        # xyz -= center

        # print(torch.mean(xyz, axis=0))
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda()

        pred_sdf = decoder(inputs)

        # TODO: why is this needed?
        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        #loss = loss_l1(pred_sdf, sdf_gt)
        loss = loss_l2(pred_sdf, sdf_gt)

        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.cpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


def list_depth_frames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                
                pose_path = os.path.join(data_source,class_name,instance_name,'tf/tf_allposes.npz')
                poses = np.load(pose_path)['arr_0']
                max_frame_id = len(poses)

                base_path = os.path.join(data_source,class_name,instance_name,'realsense/depth/')
                base_dir = os.listdir(base_path)
                base_dir.sort()

                for frame_id, frame in enumerate(base_dir):
                    if frame_id == max_frame_id: break
                    npzfiles += [os.path.join(base_path,frame)]
    return npzfiles

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to reconstruct a shape given SDF "
        + "samples."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to reconstruct.",
    )
    arg_parser.add_argument(
        "--iters",
        dest="iterations",
        default=800,
        help="The number of iterations of latent code optimization to perform.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        action="store_true",
        help="Skip meshes which have already been reconstructed.",
    )
    arg_parser.add_argument(
        "--partial",
        dest="partial",
        action="store_true",
        help="Use only partial clouds for recontructions.",
    )
    arg_parser.add_argument(
        "--depth",
        dest="depth",
        action="store_true",
        help="Use depth for recontructions.",
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()
    
    deep_sdf.configure_logging(args)

    def load_latents(exp_dir):
        indices = []
        latents = []
        latents_trained = torch.load(exp_dir+'/LatentCodes/latest.pth')
        for idx, l in enumerate(latents_trained['latent_codes']['weight']):
            latents.append(torch.clone(l))
            indices.append(idx)

        return latents, torch.tensor(indices)     

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            current_lat = latent_vecs[ind]
            lat_mat = torch.cat([lat_mat, current_lat.cuda()], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        print('stats: ', mean, var)
        return mean, var

    lat_vecs_trained, indices_trained = load_latents(args.experiment_directory)
    emp_mean, emp_var = empirical_stat(lat_vecs_trained, indices_trained)

    specs_filename = os.path.join(args.experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("deepsdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(
            args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"
        )
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    with open(args.split_filename, "r") as f:
        split = json.load(f)

    if args.depth:
        npz_filenames = list_depth_frames(args.data_source, split)
    else:
        npz_filenames = deep_sdf.data.get_instance_filenames(args.data_source, split)


    logging.debug(decoder)

    err_sum = 0.0
    repeat = 1
    save_latvec_only = False
    rerun = 0

    reconstruction_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(saved_model_epoch)
    )

    if not os.path.isdir(reconstruction_dir):
        os.makedirs(reconstruction_dir)

    reconstruction_meshes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_meshes_subdir
    )
    if args.depth:
        reconstruction_meshes_dir = os.path.join(reconstruction_meshes_dir,'depth/')
    elif args.partial:
        reconstruction_meshes_dir = os.path.join(reconstruction_meshes_dir,'partial/')
    else:
        reconstruction_meshes_dir = os.path.join(reconstruction_meshes_dir,'complete/')
    if not os.path.isdir(reconstruction_meshes_dir):
        os.makedirs(reconstruction_meshes_dir)

    reconstruction_codes_dir = os.path.join(
        reconstruction_dir, ws.reconstruction_codes_subdir
    )
    if args.depth:
        reconstruction_codes_dir = os.path.join(reconstruction_codes_dir,'depth/')
    elif args.partial:
        reconstruction_codes_dir = os.path.join(reconstruction_codes_dir,'partial/')
    else:
        reconstruction_codes_dir = os.path.join(reconstruction_codes_dir,'complete/')
    if not os.path.isdir(reconstruction_codes_dir):
        os.makedirs(reconstruction_codes_dir)

    # modifying original code here
    random.seed(0)
    random.shuffle(npz_filenames)

    total_rec_time = 0

    # TODO: testing only on some scans for now
    # npz_filenames = npz_filenames[:50]
    for ii, npz in enumerate(npz_filenames):

        print('\n', ii, '\n')

        if args.depth:
            if "npy" not in npz:
                continue

        else:
            if "npz" not in npz:
                continue

        full_filename = os.path.join(args.data_source, ws.sdf_samples_subdir, npz)

        logging.debug("loading {}".format(npz))

        if args.depth:
            data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename, depth=args.depth)
        else:
            data_sdf = deep_sdf.data.read_sdf_samples_into_ram(full_filename, partial=args.partial)


        results_filename = npz[:-4]
        if args.depth:
            results_filename = npz.split('/')[-4] + '_' + npz.split('/')[-1][:-4]
        else:
            results_filename = npz.split('/')[-3]

        for k in range(repeat):

            if rerun > 1:
                mesh_filename = os.path.join(
                    reconstruction_meshes_dir, results_filename + "-" + str(k + rerun)
                )
                latent_filename = os.path.join(
                    reconstruction_codes_dir, results_filename + "-" + str(k + rerun) + ".pth"
                )
            else:
                mesh_filename = os.path.join(reconstruction_meshes_dir, results_filename)
                latent_filename = os.path.join(
                    reconstruction_codes_dir, results_filename + ".pth"
                )

            if (
                args.skip
                and os.path.isfile(mesh_filename + ".ply")
                and os.path.isfile(latent_filename)
            ):
                continue

            logging.info("reconstructing {}".format(npz))

            data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
            data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

            start = time.time()
            err, latent = reconstruct(
                decoder,
                int(args.iterations),
                latent_size,
                data_sdf,
                [emp_mean,emp_var],
                1,
                num_samples=32000,
                lr=0.1,
                l2reg=False,
            )
            # total_rec_time += time.time() - start
            logging.info("reconstruct time: {}".format(time.time() - start))
            err_sum += err
            logging.debug("current_error avg: {}".format((err_sum / (ii + 1))))
            logging.debug(ii)

            logging.debug("latent: {}".format(latent.detach().cpu().numpy()))

            decoder.eval()
            print('latent final:', latent)
            if not args.depth:
                if not os.path.exists(os.path.dirname(mesh_filename)):
                    os.makedirs(os.path.dirname(mesh_filename))

            if not save_latvec_only:
                logging.info("saving: {}".format(mesh_filename))
                with torch.no_grad():
                    # add file name to decide where to save
                    rec_sampling_time = deep_sdf.mesh.create_mesh(
                        decoder, latent, mesh_filename, start=start, N=256, max_batch=int(2 ** 18)
                    )

                # logging.debug("total time: {}".format(time.time() - start))

            total_rec_time += rec_sampling_time

            if not os.path.exists(os.path.dirname(latent_filename)):
                os.makedirs(os.path.dirname(latent_filename))

            torch.save(latent.unsqueeze(0), latent_filename)
        
        
    print()
    print()
    print('average reconstruction time: ', total_rec_time/len(npz_filenames))
