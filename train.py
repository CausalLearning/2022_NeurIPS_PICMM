from options.train_options import TrainOptions
from dataloader.data_loader import dataloader
from model import create_model
import os
import time
import datetime


if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    # create a dataset
    dataset = dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('training images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    # training flag
    keep_training = True
    max_iteration = opt.niter + opt.niter_decay
    epoch = 0
    total_iteration = opt.iter_count
    global_time = time.time()
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    # training process
    while(keep_training):
        epoch_start_time = time.time()
        epoch += 1
        print('\n Training epoch: %d' % epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iteration += 1
            model.set_input(data)
            model.optimize_parameters()

            # save images created by PICMM of every <display_freq> iterations
            if total_iteration % opt.display_freq == 0 or total_iteration == opt.iter_count + 1:
                visual_ret = model.get_current_visuals()
                img_file = os.path.join(opt.save_img_dir, opt.name)
                img_file_gmm = os.path.join(opt.save_gmm_img, opt.name)
                os.makedirs(img_file, exist_ok=True)
                os.makedirs(img_file_gmm, exist_ok=True)

                img_name_g = "epoch%d_iter%d_g.png" % (epoch, total_iteration)
                img_file_g = os.path.join(img_file, img_name_g)
                sample = visual_ret['img_g'][-1]
                model.save_img(sample, img_file_g)

                img_name_out = "epoch%d_iter%d_out.png" % (epoch, total_iteration)
                img_file_out = os.path.join(img_file, img_name_out)
                sample_out = visual_ret['img_out']
                model.save_img(sample_out, img_file_out)

                for kk in range(opt.k):
                    img_name_gmm = "epoch%d_iter%d_K%d.png" % (epoch, total_iteration, kk)
                    img_file_gmmK = os.path.join(img_file_gmm, img_name_gmm)
                    sample_gmm = model.testGMM(kk)
                    model.save_img(sample_gmm, img_file_gmmK)

            # print training loss and save logging information to the disk
            if total_iteration % opt.print_freq == 0 or total_iteration == opt.iter_count + 1:
                losses = model.get_current_errors()
                elapsed = time.time() - global_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                message = 'Elapsed: [%s] (epoch: %d, iters: %d) ' % (elapsed, epoch, total_iteration)
                for k, v in losses.items():
                    message += '%s: %.3f ' % (k, v)

                print(message)
                with open(log_name, "a") as log_file:
                    log_file.write('%s\n' % message)

            # save the model every <save_iter_freq> iterations to the disk
            if total_iteration % opt.save_iters_freq == 0 or total_iteration == opt.iter_count + 1:
                print('saving the model of iterations %d' % total_iteration)
                model.save_networks(total_iteration)

            if total_iteration > max_iteration:
                keep_training = False
                break

        model.update_learning_rate()

        print('\nEnd training')
