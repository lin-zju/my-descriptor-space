import datetime
import logging
import time

import torch
import torch.distributed as dist
from ..utils.metric_logger import MetricLogger
from .eval import test


def train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_arguments,
    params,
    
    dataloader_val=None,
    evaluator=None
):
    """
    Model training engine.
    :param model: model(data, targets) should return a loss dictionary
    :param checkpointer: Checkpointer object
    :param checkpoint_arguments: arguments that should be saved in checkpoint
    :param params: training parameters:
                  max_epochs: maximium epochs
                  checkpoint_periods: how many checkpoints
                  print_every: report every ? iterations
    :param dataloader_val: validation dataset
    :param evaluator: Evaluator object
    """

    # get arguments
    start_epoch = checkpoint_arguments['epoch']
    start_iter = checkpoint_arguments['iteration']
    max_epochs = params['max_epochs']
    checkpoint_period = params['checkpoint_period']
    print_every = params['print_every']
    max_iter = max_epochs * len(data_loader)

    # metric logger
    meters = MetricLogger(", ")
    
    print("Start training")
    
    start_training_time = time.time()
    # end: the end time of last iteration
    end = time.time()

    first = True
    for epoch in range(start_epoch, max_epochs):
        checkpoint_arguments['epoch'] = epoch
        model.train()
        
        # starting from where we drop
        enumerator = enumerate(data_loader, start_iter if first else 0)
        for iteration, (data, targets) in enumerator:
            # this is necessary to ensure the right number of epochs
            if iteration >= max_iter:
                break
                
            # time used for loading data
            data_time = time.time() - end
            iteration = iteration + 1
            
            # iteration should be kept in the checkpointer
            checkpoint_arguments['iteration'] = iteration
            
            # step learning rate scheduler
            if scheduler:
                scheduler.step()
            
            # batch training
            # put data to device
            data = data.to(device)
            targets = targets.to(device)
            
            # get losses
            loss_dict = model(data, targets)
            
            # sum all losses for bp
            meters.update(**loss_dict)
            loss = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # time for one iteration
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            
            # estimated seconds is number of iterations left / time per iteration
            eta_seconds = meters.time.global_avg * (max_iter - epoch * len(data_loader) + iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            if iteration % print_every == 0 or iteration == max_iter:
                print(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memeory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                    )
                )
                
            # save model, optimizer, scheduler, and other arguments
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:05d}_{:07d}".format(epoch, iteration), **checkpoint_arguments)
                
        # evaluate result after each epoch
        if not evaluator is None and not dataloader_val is None:
            results = test(model, device, dataloader_val, evaluator)
            print(**results)
            
            
            
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    
    print("Total training time: {} ({:.4f} s /it)".format(
        total_time_str, total_training_time / (max_iter)
    ))
