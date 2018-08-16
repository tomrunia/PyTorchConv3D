# MIT License
# 
# Copyright (c) 2018 Tom Runia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-XX-XX

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def train_epoch(config, model, data_loader):

    model.train()

    for step, (clips, targets) in enumerate(data_loader):

        start_time = time.time()

        # Prepare for next iteration
        optimizer.zero_grad()

        # Move inputs to GPU memory
        clips   = clips.to(device)
        targets = targets.to(device)

        # Feed-forward through the network
        logits = model.forward(clips)

        # Only for ResNet etc. [n_batch,400] => [n_batch,400,1]
        #if logits.dim() == 2:
        #   logits = logits.unsqueeze(-1)

        _, preds = torch.max(logits, 1)
        loss = criterion(logits, targets)

        # Calculate accuracy
        correct = torch.sum(preds == targets.data)
        accuracy = correct.double() / config.batch_size

        # Calculate elapsed time for this step
        step_time = config.batch_size/float(time.time() - start_time)

        if phase == 'train':
            loss.backward()
            optimizer.step()

        if phase == 'train':

            global_step = (epoch*train_steps_per_epoch) + step

            # Update average statistics
            examples_per_second.push(step_time)
            losses.push(loss.item())
            accuracies.push(accuracy.item())

            if step % config.print_frequency == 0:
                print("[{}] Epoch {}. Batch {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                          "LR = {:.3f}, Accuracy = {:.3f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%A %H:%M"), epoch+1, step,
                        train_steps_per_epoch, examples_per_second.average(),
                        current_learning_rate(optimizer), accuracies.average(), loss))

            if step % config.log_frequency == 0:
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/accuracy', accuracy, global_step)
                writer.add_scalar('train/examples_per_second', step_time, global_step)
                writer.add_scalar('train/learning_rate', current_learning_rate(optimizer), global_step)
                writer.add_scalar('train/weight_decay', current_weight_decay(optimizer), global_step)

        else:

            print("[{}] Epoch {}. Validation Batch {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), epoch+1, step,
                    len(datasets[phase])//config.batch_size,
                    step_time, accuracy.item(), loss.item()))

    if phase == 'train':
        scheduler.step(epoch)

    if epoch % config.checkpoint_frequency == 0:
        save_checkpoint_path = os.path.join(config.checkpoint_dir, 'save_{:03}.pth'.format(epoch))
        print('Checkpoint written to: {}'.format(save_checkpoint_path))
        save_checkpoint(save_checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())

