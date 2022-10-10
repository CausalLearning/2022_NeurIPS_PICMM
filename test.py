from options import test_options
from dataloader import data_loader
from model import create_model
import time

if __name__=='__main__':
    # get testing options
    opt = test_options.TestOptions().parse()
    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    global_time = time.time()

    for i, data in enumerate(dataset):
        print("batch %d" % (i+1))
        model.set_input(data)
        model.test()
    
    ela = time.time() - global_time
    print("total test time:", ela)