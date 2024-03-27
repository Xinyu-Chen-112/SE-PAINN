import os
import sys
import csv
import time
import json
import math
import random
import logging
import argparse
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

from tools import split_dataset, EarlyStopping, AverageMeter


def main():
    input_dir = os.path.join(args.root_dir, "input")
    output_dir = os.path.join(args.root_dir, "output")

    if args.task == "train":
        out_dir = os.path.join(output_dir, "train")
    elif args.task == "predict":
        out_dir = os.path.join(output_dir, "predict")

    # set logging
    if args.log == 10:
        level = logging.DEBUG
    elif args.log == 20:
        level = logging.INFO
    elif args.log == 30:
        level = logging.WARNING
    elif args.log == 40:
        level = logging.ERROR
    elif args.log == 50:
        level = logging.CRITICAL
    else:
        level = logging.NOTSET
    logging.basicConfig(level=level,
                        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
                        handlers=[logging.FileHandler(os.path.join(out_dir, "printlog.txt"), mode="w"),
                                  logging.StreamHandler()])

    # set random seed
    if args.random_seed:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True

    # Save command line args
    with open(os.path.join(out_dir, "commandline_args.txt"), "w") as f:
        f.write("\n".join(sys.argv[1:]))
    # Save parsed command line arguments
    with open(os.path.join(out_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f)

    # create device
    device = torch.device(args.device)
    
    # setup dataset
    if args.task == "train":
        pkl = True
        data_dir = os.path.join(input_dir, "train_data")
    else:
        pkl = False
        data_dir = os.path.join(input_dir, "predict_data")

    logging.info(f"loading data from {data_dir}")
    if args.model in ["cgcnn", "cgcnn_01"]:
        dataset = Data(pkl, data_dir,
                       cutoff=args.cutoff, max_num_nbrs=args.max_num_nbrs)
    elif args.model == "cgcnn_cv":
        dataset = Data(pkl, data_dir,
                       che_cutoff=args.che_cutoff, che_max_num_nbrs=args.che_max_num_nbrs,
                       vdw_cutoff=args.vdw_cutoff, vdw_max_num_nbrs=args.vdw_max_num_nbrs)
    elif args.model in ["megnet", "megnet_01", "painn", "painn_01"]:
        dataset = Data(pkl, data_dir, add_same=args.add_same,
                       cutoff=args.cutoff, max_num_nbrs=args.max_num_nbrs)
    elif args.model in ["megnet_cv", "painn_cv"]:
        dataset = Data(pkl, data_dir, add_same=args.add_same,
                       che_cutoff=args.che_cutoff, che_max_num_nbrs=args.che_max_num_nbrs,
                       vdw_cutoff=args.vdw_cutoff, vdw_max_num_nbrs=args.vdw_max_num_nbrs)

    if args.task == "train":
        # setup dataloader
        splited_dataset = split_dataset(dataset, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                                        test_ratio=args.test_ratio, test=args.test,
                                        save_path=os.path.join(out_dir, 'split_data.json'))

        if args.test:
            logging.info('Dataset size: {}, train set size: {}, validation set size: {}, test set size: {}'
                         .format(len(dataset), len(splited_dataset["train"]), len(splited_dataset["val"]), len(splited_dataset["test"])))
        else:
            logging.info('Dataset size: {}, train set size: {}, validation set size: {}'
                         .format(len(dataset), len(splited_dataset["train"]), len(splited_dataset["val"])))

        train_loader = DataLoader(splited_dataset["train"], batch_size=args.batch_size,
                                  sampler=RandomSampler(splited_dataset["train"]),
                                  drop_last=True, collate_fn=collate_batch, pin_memory=args.pin_memory)
        val_loader = DataLoader(splited_dataset["val"], batch_size=args.batch_size,
                                drop_last=False, collate_fn=collate_batch, pin_memory=args.pin_memory)
        if args.test:
            test_loader = DataLoader(splited_dataset["test"], batch_size=args.batch_size,
                                     drop_last=False, collate_fn=collate_batch, pin_memory=args.pin_memory)

        # setup mean, stddev
        target_mean = 0.0
        target_stddev = 1.0

        if args.normalization:  # target Z-score standardization
            n = len(splited_dataset["train"])
            tsum = 0
            t2 = 0
            for data in splited_dataset["train"]:
                t = data["target"].item()
                tsum += t
                t2 += t ** 2.0
            target_mean = tsum / n
            var = t2 / n - target_mean ** 2.0
            target_stddev = math.sqrt(var)
            logging.info("target_mean=%f, target_stddev=%f" % (target_mean, target_stddev))

        # setup model
        if args.model in ["cgcnn", "cgcnn_01"]:
            net = Model(cutoff=args.cutoff,
                        node_input_size=args.node_input_size,
                        edge_embedding_size=args.edge_embedding_size,
                        hidden_size=args.hidden_size,
                        n_conv=args.n_conv,
                        h_size=args.h_size,
                        n_h=args.n_h,
                        classification=args.classification,
                        normalization=args.normalization,
                        target_mean=target_mean,
                        target_stddev=target_stddev)
        elif args.model == "cgcnn_cv":
            net = Model(che_cutoff=args.che_cutoff,
                        vdw_cutoff=args.vdw_cutoff,
                        node_input_size=args.node_input_size,
                        edge_embedding_size=args.edge_embedding_size,
                        hidden_size=args.hidden_size,
                        n_conv=args.n_conv,
                        h_size=args.h_size,
                        n_h=args.n_h,
                        classification=args.classification,
                        normalization=args.normalization,
                        target_mean=target_mean,
                        target_stddev=target_stddev)
        elif args.model in ["megnet", "megnet_01"]:
            net = Model(cutoff=args.cutoff,
                        edge_embedding_size=args.edge_embedding_size,
                        node_input_size=args.node_input_size,
                        node_embedding_size=args.node_embedding_size,
                        state_input_size=args.state_input_size,
                        inner_skip=args.inner_skip,
                        hidden_size=args.hidden_size,
                        n_blocks=args.n_blocks,
                        pool_method=args.pool_method,
                        n_set2set=args.n_set2set,
                        normalization=args.normalization,
                        target_mean=target_mean,
                        target_stddev=target_stddev)
        elif args.model == "megnet_cv":
            net = Model(che_cutoff=args.che_cutoff,
                        vdw_cutoff=args.vdw_cutoff,
                        edge_embedding_size=args.edge_embedding_size,
                        node_input_size=args.node_input_size,
                        node_embedding_size=args.node_embedding_size,
                        state_input_size=args.state_input_size,
                        inner_skip=args.inner_skip,
                        hidden_size=args.hidden_size,
                        n_blocks=args.n_blocks,
                        pool_method=args.pool_method,
                        n_set2set=args.n_set2set,
                        normalization=args.normalization,
                        target_mean=target_mean,
                        target_stddev=target_stddev)
        elif args.model in ["painn", "painn_01"]:
            net = Model(num_interactions=args.num_interactions,
                        cutoff=args.cutoff,
                        node_input_size=args.node_input_size,
                        edge_embedding_size=args.edge_embedding_size,
                        hidden_size=args.hidden_size,
                        normalization=args.normalization,
                        atomwise_normalization=args.atomwise_normalization,
                        target_mean=target_mean,
                        target_stddev=target_stddev)
        elif args.model == "painn_cv":
            net = Model(num_interactions=args.num_interactions,
                        che_cutoff=args.che_cutoff,
                        vdw_cutoff=args.vdw_cutoff,
                        node_input_size=args.node_input_size,
                        edge_embedding_size=args.edge_embedding_size,
                        hidden_size=args.hidden_size,
                        normalization=args.normalization,
                        atomwise_normalization=args.atomwise_normalization,
                        target_mean=target_mean,
                        target_stddev=target_stddev)

        net.to(device)

        # setup optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=args.initial_lr)
        criterion = torch.nn.MSELoss()

        if args.plateau_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_factor, patience=args.lr_patience)
        else:
            scheduler_fn = lambda last_epoch: 0.96 ** (last_epoch / 10)
            scheduler = LambdaLR(optimizer, scheduler_fn)

        # setup early stop
        early_stop = EarlyStopping(escape=args.stop_escape, threshold=0, patience=args.stop_patience)

        step = 1
        
        # used for smoothing loss
        prev_loss = None
        best_val_loss = np.inf

        loss_csv_dir = os.path.join(out_dir, "loss.csv")
        if os.path.exists(loss_csv_dir):
            os.remove(loss_csv_dir)

        if args.load_model:
            logging.info(f"Load model from {args.load_model}")
            state_dict = torch.load(args.load_model)
            net.load_state_dict(state_dict["model"])
            step = state_dict["step"]
            best_val_loss = state_dict["best_val_loss"]
            optimizer.load_state_dict(state_dict["optimizer"])
            scheduler.load_state_dict(state_dict["scheduler"])
            with open(loss_csv_dir, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(state_dict["loss"])

        train_loss_meter = AverageMeter()
        train_time_meter = AverageMeter()
        val_time_meter = AverageMeter()

        all_loss = []

        epoch = True
        while epoch:
            for train_batch in train_loader:
                train_start = time.time()
                net.train()
                # Transfer to device
                train_batch_device = {k: v.to(device=device, non_blocking=True) for (k, v) in train_batch.items()}
                # Reset gradient
                optimizer.zero_grad()
                # Forward
                train_out = net(train_batch_device)
                # Loss
                train_loss = criterion(train_out, train_batch_device["target"])
                train_loss = torch.sqrt(train_loss)
                # Backward -> Calculate gradients
                train_loss.backward()
                # Optimize-> Update parameters
                optimizer.step()
                # Averagemeter
                train_loss_meter.update(train_loss.item(), train_out.shape[0])
                train_time_meter.update(time.time() - train_start)

                logging.info("step {:4d} : train_time={:.3f} s, total={:.3f} s, mean={:.3f} s"
                             .format(step, train_time_meter.val, train_time_meter.sum, train_time_meter.avg))

                # Validate and save model
                if (step % args.log_interval == 0) or (step == args.max_steps):
                    logging.info("Train Interval Report : train_rmse_mean={:.6f}".format(train_loss_meter.avg))

                    # Validate
                    val_start = time.time()
                    net.eval()
                    with torch.no_grad():
                        val_outs = torch.tensor([], device=device)
                        val_targets = torch.tensor([], device=device)
                        for val_batch in val_loader:
                            # Transfer to device
                            val_batch_device = {k: v.to(device=device, non_blocking=True) for k, v in val_batch.items()}
                            # Forward
                            val_out = net(val_batch_device)
                            # Splicing
                            val_outs = torch.cat((val_outs, val_out), dim=0)
                            val_targets = torch.cat((val_targets, val_batch_device["target"]), dim=0)

                    # Loss
                    val_loss = criterion(val_outs, val_targets)
                    val_loss = torch.sqrt(val_loss)

                    val_time_meter.update(time.time() - val_start)
                    logging.info("\t\tValidate : val_time={:.3f} s, total={:.3f} s, mean={:.3f} s"
                                 .format(val_time_meter.val, val_time_meter.sum, val_time_meter.avg))
                    
                    # loss smoothing
                    smooth_loss = val_loss if prev_loss is None else args.pre_weight * prev_loss + (1 - args.pre_weight) * val_loss
                    logging.info("\t\t\tval_rmse={:.6f}, smooth_rmse={:.6f}"
                                 .format(val_loss.item(), smooth_loss.item()))

                    loss = [step, train_loss_meter.avg, smooth_loss.item()]
                    all_loss.append(loss)
                    with open(loss_csv_dir, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows([loss])

                    # Reset Averagemeter
                    train_loss_meter.reset()

                    prev_loss = smooth_loss

                    step += 1

                    # reduce learning rate
                    if args.plateau_scheduler:
                        scheduler.step(smooth_loss)
                    else:
                        scheduler.step()

                    # Save checkpoint
                    if smooth_loss < best_val_loss:
                        torch.save({"net": net,
                                    "model": net.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    "step": step,
                                    "best_val_loss": smooth_loss,
                                    "loss": all_loss},
                                   os.path.join(out_dir, "best_model.pth"))

                    torch.save({"net": net,
                                "model": net.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "step": step,
                                "best_val_loss": min(best_val_loss, smooth_loss),
                                "loss": all_loss},
                               os.path.join(out_dir, "last_model.pth"))

                    if early_stop(smooth_loss, best_val_loss):
                        logging.warning("Early Stopping")
                        epoch = False
                        break
                    else:
                        best_val_loss = min(best_val_loss, smooth_loss)
                else:
                    step += 1
                    if not args.plateau_scheduler:
                        scheduler.step()

                if step > args.max_steps:
                    logging.warning("Max Steps Reached")
                    epoch = False
                    break

        # plot loss
        data = np.genfromtxt(loss_csv_dir, delimiter=',', encoding='utf-8-sig')
        plt.plot(data[:, 0], data[:, 1], color='red', linestyle='-', linewidth=2)
        plt.plot(data[:, 0], data[:, 2], color='blue', linestyle='-', linewidth=2)
        plt.xlabel('step', fontsize=15)
        plt.ylabel('RMSE', fontsize=15)
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.gcf().subplots_adjust(left=0.2)
        plt.legend(['train rmse', 'val rmse'], frameon=False, loc="upper right", fontsize='small', ncol=1,
                   prop={'family': 'Times New Roman', 'weight': 'normal',
                         'size': 20})
        plot_save_path = os.path.join(out_dir, 'rmse-loss.png')
        plt.savefig(plot_save_path)
        plt.clf()

    # predict
    elif args.task == "predict":
        train_dir = os.path.join(output_dir, "train")

        pth_dir = os.path.join(train_dir, "best_model.pth") if args.choose == "best" else os.path.join(train_dir, "last_model.pth")
        # load model
        state_dict = torch.load(pth_dir)
        net = state_dict["net"]
        net.load_state_dict(state_dict["model"])
        net.to(device)

        predict_loader = DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=False, drop_last=False, collate_fn=collate_batch, pin_memory=args.pin_memory)

        net.eval()
        with torch.no_grad():
            predict_outs = torch.tensor([], device=device)
            predict_targets = torch.tensor([], device=device)
            for predict_batch in predict_loader:
                # Transfer to device
                predict_batch_device = {k: v.to(device=device, non_blocking=True) for k, v in predict_batch.items()}
                # Forward
                predict_out = net(predict_batch_device)
                # Splicing
                predict_outs = torch.cat((predict_outs, predict_out), dim=0)
                predict_targets = torch.cat((predict_targets, predict_batch_device["target"]), dim=0)

        predictions = np.column_stack((np.array(predict_targets.cpu()), np.array(predict_outs.cpu())))

        samples_predict_dir = os.path.join(out_dir, "samples-predict.csv")
        if os.path.exists(samples_predict_dir):
            os.remove(samples_predict_dir)
        with open(samples_predict_dir, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parameters for graph convolution networks", fromfile_prefix_chars="@")

    # system args
    parser.add_argument("root_dir", type=str, help="Path to root directory")
    parser.add_argument("model", type=str, choices=["cgcnn", "cgcnn_01", "cgcnn_cv",
                                                    "megnet", "megnet_01", "megnet_cv",
                                                    "painn", "painn_01", "painn_cv"], help="Model to use")
    parser.add_argument("task", type=str, choices=["train", "predict"], help="train or predict")

    parser.add_argument("--log", default=20, type=int, choices=[10, 20, 30, 40, 50], help="Rank to logging out")
    parser.add_argument("--random_seed", default=None, type=int, help="Random seed for this run")
    parser.add_argument("--device", default='cpu', type=str, help="Set which device to use for training e.g. 'cuda' or 'cpu'")
    parser.add_argument("--pin_memory", default=False, action="store_true", help="Enable DataLoader's pin_memory")

    # for train
    parser.add_argument("--load_model", default=None, type=str, help="The path to load model from previous run")
    parser.add_argument("--train_ratio", default=None, type=float, help="Ratio of train set")
    parser.add_argument("--val_ratio", default=0.1, type=float, help="Ratio of validation set")
    parser.add_argument("--test_ratio", default=0.1, type=float, help="Ratio of test set")
    parser.add_argument("--test", default=False, action="store_true", help="whether return test set")
    parser.add_argument("--batch_size", default=16, type=int, help="Number of structures per minibatch")
    parser.add_argument("--add_same", default=False, action="store_true", help="whether add neighbors with same distance")

    parser.add_argument("--initial_lr", default=0.005, type=float, help="Initial learning rate")
    parser.add_argument("--plateau_scheduler", default=False, action="store_true", help="Using ReduceLROnPlateau scheduler for decreasing learning rate when learning plateaus")
    parser.add_argument("--lr_factor", default=0.5, type=float, help="ReduceLROnPlateau scheduler's decrease factor, ONLY for --plateau_scheduler")
    parser.add_argument("--lr_patience", default=5, type=int, help="ReduceLROnPlateau scheduler's decrease patience, ONLY for --plateau_scheduler")
    
    parser.add_argument("--stop_escape", default=100, type=int, help="Early Stopping does not consider the first 'stop_escape' times")
    parser.add_argument("--stop_patience", default=20, type=int, help="Stop training when validation loss is larger than best loss for 'stop_patience' steps")
    parser.add_argument("--log_interval", default=5, type=int, help="The interval of model evaluation")
    parser.add_argument("--max_steps", default=500, type=int, help="Maximum number of optimisation steps")
    parser.add_argument("--pre_weight", default=None, type=float, help="smooth_loss = pre_weight*prev_loss + (1-pre_weight)*val_loss")

    # for None and 01
    parser.add_argument("--cutoff", default=5.0, type=float, help="Atomic interaction cutoff distance (Å)")
    parser.add_argument("--max_num_nbrs", default=12, type=int, help="The maximum number of neighbors considered")
    # for cv
    parser.add_argument("--che_cutoff", default=5.0, type=float, help="Intralayer Atomic interaction cutoff distance (Å)")
    parser.add_argument("--che_max_num_nbrs", default=12, type=int, help="The maximum number of intralayer neighbors considered")
    parser.add_argument("--vdw_cutoff", default=5.0, type=float, help="Interlayer Atomic interaction cutoff distance (Å)")
    parser.add_argument("--vdw_max_num_nbrs", default=12, type=int, help="The maximum number of interlayer neighbors considered")

    parser.add_argument("--node_input_size", default=13, type=int, help="Size of origin node")
    parser.add_argument("--edge_embedding_size", default=20, type=int, help="Size of RBF of distance of edge")
    parser.add_argument("--hidden_size", default=128, type=int, help="Size of hidden node states")
    parser.add_argument("--normalization", default=False, action="store_true", help="Enable normalization of the model")
    # for cgcnn
    parser.add_argument("--n_conv", default=3, type=int, help="Number of ConvLayer used")
    parser.add_argument("--h_size", default=128, type=int, help="Size after pool")
    parser.add_argument("--n_h", default=1, type=int, help="Number of FullConnectLayer after pool")
    parser.add_argument("--classification", default=False, action="store_true", help="Predict for classification or regression")
    # for megnet
    parser.add_argument("--node_embedding_size", default=16, type=int, help="Size of embedding node")
    parser.add_argument("--state_input_size", default=1, type=int, help="Size of input state")
    parser.add_argument("--inner_skip", default=False, action="store_true", help="Use inner or outer skip connection")
    parser.add_argument("--n_blocks", default=3, type=int, help="Number of MegnetBlock used")
    parser.add_argument("--pool_method", default="sum", type=str, choices=["sum", "mean"], help="sum pool or mean pool")
    parser.add_argument("--n_set2set", default=1, type=int, help="Number of Set2Set's processing_steps")
    # for painn
    parser.add_argument("--num_interactions", default=3, type=int, help="Number of interaction layers used")
    parser.add_argument("--atomwise_normalization", default=False, action="store_true", help="Enable normalization's atomwise normalization")

    # for predict
    parser.add_argument("--choose", default="last", choices=["best", "last"],
                        help="predict with best model or last model")

    args = parser.parse_args(sys.argv[1:])

    exec(f"from data.{args.model} import Data, collate_batch")
    exec(f"from model.{args.model} import Model")

    main()
