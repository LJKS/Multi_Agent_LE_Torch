import torch
import data
import agents
import training
import torch.nn.functional as F
import torchmetrics
import higher
import numpy as np
import commentary_networks
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import random

import utils

TEST_VAL_SPLIT = 0.5 #of the designated 'test' data we treat the first 'TEST_VAL_SPLIT' as test data by default, the rest as validation
#copied utils from training
def target_distractor_encode_data(features_batch, target_idx_batch, num_classes):
    onehot_encoding = F.one_hot(target_idx_batch, num_classes=num_classes)
    onehot_encoding = torch.swapaxes(onehot_encoding, 1,2) # (batchsize, 1, numtargets+distracts) TO (batchsize, numtargets+distracts, 1)
    target_encoded_features = torch.cat((onehot_encoding, features_batch), dim=2)
    return  target_encoded_features


def prob_mask(tokens, eos_token=4):
    #only include timesteps before and including endofsequence
    #creates a mask that is 0 for all elements after the eostoken has been reached, 1 else

    #check where you find eos tokens:
    eos_tokens = tokens==eos_token

    #mark everything that is or is after a eos token
    eos_tokens = torch.cumsum(eos_tokens, dim=1)
    #do it twice so the first eos token has a value of 1, all later tokens are > 1
    eos_tokens = torch.cumsum(eos_tokens, dim=1)


    #now we have counts but we want whether count is <=1 as a float (to include the eos token!)

    eos_tokens = (eos_tokens<=1.).to(dtype=float)

    return eos_tokens

def supervised_receiver_test(receiver, criterion, test_loss, test_acc, target_idx_batch, all_features_batch, target_captions_stack):
    target_idx_batch = torch.squeeze(target_idx_batch)

    logits = receiver(all_features_batch, target_captions_stack)
    loss = torch.mean(criterion(logits, target_idx_batch))

    test_acc.update(logits.to('cpu'), target_idx_batch.to('cpu'))
    test_loss.update(loss.to('cpu'))

def supervised_sender_test(sender, criterion, test_loss, test_pp, target_encoded_features, target_captions_stack, vocab, write_flag=False, num_writes=5, writer=None, sentence_tag='unk', episode=None, device=None):
    logits = sender(target_encoded_features, target_captions_stack[:, :-1], device=device)

    if write_flag:
        write_logits = logits[0:num_writes, :, :]
        argmax_seqs = torch.argmax(write_logits, dim=-1).cpu().detach()
        test_sentences = utils.batch2sentences(vocab, argmax_seqs)
        org_sentences = utils.batch2sentences(vocab, target_captions_stack.cpu().detach())
        write_string = utils.summarize_test_sentences(org_sentences, test_sentences)
        writer.add_text(tag=sentence_tag, text_string=write_string, global_step=episode)

    loss = criterion(torch.swapaxes(logits, 1, 2), target_captions_stack[:, 1:])
    loss = loss * prob_mask(target_captions_stack)[:, 1:]
    loss = torch.mean(loss)
    pp = torch.exp(loss)
    test_loss.update(loss.to('cpu'))
    test_pp.update(pp.to('cpu'))


def target_distractor_encode_data(features_batch, target_idx_batch, num_classes):
    onehot_encoding = F.one_hot(target_idx_batch, num_classes=num_classes)
    onehot_encoding = torch.swapaxes(onehot_encoding, 1,2) # (batchsize, 1, numtargets+distracts) TO (batchsize, numtargets+distracts, 1)
    target_encoded_features = torch.cat((onehot_encoding, features_batch), dim=2)
    return  target_encoded_features


def prob_mask(tokens, eos_token=4):
    #only include timesteps before and including endofsequence
    #creates a mask that is 0 for all elements after the eostoken has been reached, 1 else

    #check where you find eos tokens:
    eos_tokens = tokens==eos_token

    #mark everything that is or is after a eos token
    eos_tokens = torch.cumsum(eos_tokens, dim=1)

    #now we have counts but we want whether count is >0 as a float

    eos_tokens = (eos_tokens>0).to(dtype=float)

    return eos_tokens


def baseline_multiagent_training_interactive_only(senders, receivers, receiver_lr, sender_lr, num_distractors, path, writer_tag, num_episodes=200, batch_size=512, num_workers=4, repeats_per_epoch=1, device='cpu', baseline_polyak=0.99):

    #indexing here will generally be first sender index, then receiver index for anything that has num_senders x num_receivers elements arranged in a 2d grid

    num_senders = len(senders)
    num_receivers = len(receivers)

    [receiver.to(device) for receiver in receivers]
    [receiver.train() for receiver in receivers]
    [sender.to(device) for sender in senders]
    [sender.train() for sender in senders]

    optimizers_receiver = [torch.optim.Adam(receiver.parameters(), lr=receiver_lr) for receiver in receivers]
    optimizers_sender = [torch.optim.Adam(sender.parameters(), lr=sender_lr) for sender in senders]

    schedulers_receiver = [torch.optim.lr_scheduler.StepLR(optimizer_receiver, 1.0, gamma=0.95) for optimizer_receiver in optimizers_receiver]
    schedulers_sender = [torch.optim.lr_scheduler.StepLR(optimizer_sender, 1.0, gamma=0.95) for optimizer_sender in optimizers_sender]

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    baselines = torch.zeros(size=(num_senders,num_receivers)).to(device=device)

    (train_ds, test_ds), vocab = data.load_prepared_coco_data()

    data_loader_train = data.create_data_loader(
        train_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )

    #split test data in val and test data
    test_val_cutoff = int(len(train_ds)*TEST_VAL_SPLIT)
    validation_ds = test_ds[test_val_cutoff:]
    test_ds = test_ds[:test_val_cutoff]
    data_loader_test = data.create_data_loader(
        test_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    #metrics
    writer = SummaryWriter(path)
    training_loss_grid = [[torchmetrics.MeanMetric() for _ in range(num_receivers)] for _ in range(num_senders)]
    training_acc_grid = [[torchmetrics.Accuracy() for _ in range(num_receivers)] for _ in range(num_senders)]
    test_loss_grid = [[torchmetrics.MeanMetric() for _ in range(num_receivers)] for _ in range(num_senders)]
    test_acc_grid = [[torchmetrics.Accuracy() for _ in range(num_receivers)] for _ in range(num_senders)]
    entropies = [torchmetrics.MeanMetric() for _ in range(num_senders)]

    for episode in (p_bar := tqdm(range(num_episodes))):

        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in utils.repeat_dataset(data_loader_train, repeats_per_epoch):

            receiver_idx = random.randrange(0, num_receivers)
            sender_idx = random.randrange(0, num_senders)
            sender = senders[sender_idx]
            receiver = receivers[receiver_idx]
            receiver.train()
            sender.train()

            optimizer_sender = optimizers_sender[sender_idx]
            optimizer_receiver = optimizers_receiver[receiver_idx]

            optimizer_receiver.zero_grad()
            optimizer_sender.zero_grad()

            all_features_batch = all_features_batch.to(device=device)
            #target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch, num_distractors+1)#before squeezing!
            target_idx_batch = torch.squeeze(target_idx_batch)


            seq, log_p = sender(target_encoded_features, seq_data=None, device=device)
            seq = seq.detach() # technically not necessary but kind of nicer

            logits = receiver(all_features_batch, seq)
            loss = criterion(logits, target_idx_batch)

            #Receiver update (classic supervised learning)
            receiver_loss = torch.mean(loss)
            receiver_loss.backward()
            optimizer_receiver.step()

            #Sender update REINFORCE!
            log_p_mask = prob_mask(seq)
            log_p = log_p*log_p_mask
            log_p = torch.sum(log_p, dim=1)
            value = -loss.detach()
            baselined_value = value - baselines[sender_idx, receiver_idx]
            sender_reinforce_objective = log_p*baselined_value
            sender_loss = torch.mean(-sender_reinforce_objective)
            sender_loss.backward()
            optimizer_sender.step()

            avg_value = torch.mean(value)
            baselines[sender_idx, receiver_idx] = baseline_polyak*baselines[sender_idx, receiver_idx] + (1.-baseline_polyak)*avg_value

            training_loss_grid[sender_idx][receiver_idx].update(-torch.mean(value).to(device='cpu'))
            training_acc_grid[sender_idx][receiver_idx].update(logits.to(device='cpu'), target_idx_batch.to(device='cpu'))
            entropies[sender_idx].update(-torch.mean(log_p.to(device='cpu')))



        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in utils.repeat_dataset(data_loader_test, repeats_per_epoch):
            receiver_idx = random.randrange(0, num_receivers)
            sender_idx = random.randrange(0, num_senders)
            sender = senders[sender_idx]
            receiver = receivers[receiver_idx]
            receiver.eval()
            sender.eval()

            all_features_batch = all_features_batch.to(device=device)
            #target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch, num_distractors+1)#before squeezing!
            target_idx_batch = torch.squeeze(target_idx_batch)


            seq, log_p = sender(target_encoded_features, seq_data=None, device=device)
            seq = seq.detach() # technically not necessary but kind of nicer

            logits = receiver(all_features_batch, seq)
            loss = criterion(logits, target_idx_batch)

            value = -loss.detach()
            test_loss_grid[sender_idx][receiver_idx].update(-torch.mean(value).to(device='cpu'))
            test_acc_grid[sender_idx][receiver_idx].update(logits.to(device='cpu'), target_idx_batch.to(device='cpu'))

        #end of episode stuff
        [scheduler_receiver.step() for scheduler_receiver in schedulers_receiver]
        [scheduler_sender.step() for scheduler_sender in schedulers_sender]

        entropy_results = []
        training_acc_result = []
        training_loss_result = []
        test_acc_result = []
        test_loss_result = []

        write_dict = {}
        for sender_idx in range(num_senders):
            episode_entropy = entropies[sender_idx].compute()
            entropies[sender_idx].reset()
            write_dict[f'entropy_sender_{sender_idx}'] = episode_entropy
            entropy_results.append(episode_entropy)
            for receiver_idx in range(num_receivers):
                training_loss = training_loss_grid[sender_idx][receiver_idx]
                episode_training_loss = training_loss.compute()
                training_loss.reset()

                training_acc = training_acc_grid[sender_idx][receiver_idx]
                episode_training_acc = utils.safe_compute_accuracy_metric(training_acc)
                training_acc.reset()

                test_loss = test_loss_grid[sender_idx][receiver_idx]
                episode_test_loss = test_loss.compute()
                test_loss.reset()

                test_acc = test_acc_grid[sender_idx][receiver_idx]
                episode_test_acc = utils.safe_compute_accuracy_metric(test_acc)
                test_acc.reset()

                write_dict[f'training_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_loss
                write_dict[f'training_acc_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_acc
                write_dict[f'test_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_test_loss
                write_dict[f'training_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_loss

                training_acc_result.append(episode_training_acc)
                training_loss_result.append(episode_training_loss)
                test_acc_result.append(episode_test_acc)
                test_loss_result.append(episode_test_loss)

        p_bar.set_description(
            f'Train: L{np.mean(training_loss_result) :.3e} / ACC{np.mean(training_acc_result) :.3e} || Test: L{np.mean(test_loss_result) :.3e} / ACC{np.mean(test_acc_result) :.3e}')
        writer.add_scalars(main_tag=writer_tag,
                           tag_scalar_dict=write_dict,
                           global_step=episode)

    writer.flush()

def idx_commentary_training_interactive_only(senders, receivers, commentary_network, receiver_lr, sender_lr, commentary_lr, num_distractors, path, writer_tag, num_inner_loop_steps=2, num_episodes=200, batch_size=512, num_workers=4, repeats_per_epoch=1, device='cpu', baseline_polyak=0.99):
    # indexing here will generally be first sender index, then receiver index for anything that has num_senders x num_receivers elements arranged in a 2d grid

    num_senders = len(senders)
    num_receivers = len(receivers)

    [sender.to(device) for sender in senders]
    [receiver.to(device) for receiver in receivers]
    commentary_network.to(device)

    [receiver.to(device) for receiver in receivers]
    [receiver.train() for receiver in receivers]
    [sender.to(device) for sender in senders]
    [sender.train() for sender in senders]

    optimizers_receiver = [torch.optim.Adam(receiver.parameters(), lr=receiver_lr) for receiver in receivers]
    optimizers_sender = [torch.optim.Adam(sender.parameters(), lr=sender_lr) for sender in senders]
    optimizer_commentary_network = torch.optim.Adam(commentary_network.parameters(), lr=commentary_lr)

    schedulers_receiver = [torch.optim.lr_scheduler.StepLR(optimizer_receiver, 1.0, gamma=0.95) for optimizer_receiver
                           in optimizers_receiver]
    schedulers_sender = [torch.optim.lr_scheduler.StepLR(optimizer_sender, 1.0, gamma=0.95) for optimizer_sender in
                         optimizers_sender]

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    baselines = torch.zeros(size=(num_senders, num_receivers)).to(device=device)

    sender_idx_batch = [(torch.ones(size=(batch_size,),dtype=int)*idx).to(device=device) for idx in range(num_senders)]
    receiver_idx_batch = [(torch.ones(size=(batch_size,),dtype=int)*idx).to(device=device) for idx in range(num_receivers)]


    (train_ds, test_ds), vocab = data.load_prepared_coco_data()

    data_loaders_train = [data.create_data_loader(
        train_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    ) for _ in range(num_inner_loop_steps)]

    # split test data in val and test data
    test_val_cutoff = int(len(test_ds) * TEST_VAL_SPLIT)
    validation_ds = test_ds[test_val_cutoff:]
    test_ds = test_ds[:test_val_cutoff]
    data_loader_test = data.create_data_loader(
        test_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    data_loader_validation = data.create_data_loader(
        validation_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    # metrics
    writer = SummaryWriter(path)
    training_loss_grid = [[torchmetrics.MeanMetric() for _ in range(num_receivers)] for _ in range(num_senders)]
    training_acc_grid = [[torchmetrics.Accuracy() for _ in range(num_receivers)] for _ in range(num_senders)]
    test_loss_grid = [[torchmetrics.MeanMetric() for _ in range(num_receivers)] for _ in range(num_senders)]
    test_acc_grid = [[torchmetrics.Accuracy() for _ in range(num_receivers)] for _ in range(num_senders)]
    commentary_weight_grid =[[torchmetrics.MeanMetric() for _ in range(num_receivers)] for _ in range(num_senders)]

    sup_sender_loss = [torchmetrics.MeanMetric() for _ in range(num_senders)]
    sup_sender_pp = [torchmetrics.MeanMetric() for _ in range(num_senders)]
    sup_receiver_loss = [torchmetrics.MeanMetric() for _ in range(num_receivers)]
    sup_receiver_acc = [torchmetrics.Accuracy() for _ in range(num_receivers)]

    #validation is not interactive testing but
    entropies = [torchmetrics.MeanMetric() for _ in range(num_senders)]

    for episode in (p_bar := tqdm(range(num_episodes))):
        zipped_training_data_loader = zip(*data_loaders_train)
        training_val_data_loader = zip(utils.cycle_dataloader(data_loader_validation), zipped_training_data_loader)
        for val_data, inner_loop_step_data in utils.repeat_dataset(training_val_data_loader, repeats_per_epoch):

            optimizer_commentary_network.zero_grad()
            receiver_idx = random.randrange(0, num_receivers)
            sender_idx = random.randrange(0, num_senders)
            sender = senders[sender_idx]
            receiver = receivers[receiver_idx]
            receiver.train()
            sender.train()

            optimizer_sender = optimizers_sender[sender_idx]
            optimizer_receiver = optimizers_receiver[receiver_idx]
            optimizer_receiver.zero_grad()
            optimizer_sender.zero_grad()

            #inner loop
            with higher.innerloop_ctx(model=sender, opt=optimizer_sender, copy_initial_weights=False) as (
            sender_inner, optimizer_sender_inner):
                with higher.innerloop_ctx(model=receiver, opt=optimizer_receiver, copy_initial_weights=False) as (
                receiver_inner, optimizer_receiver_inner):
                    for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in inner_loop_step_data:

                        #cudnn does not support double backwards pass :(
                        with torch.backends.cudnn.flags(enabled=False):

                            all_features_batch = all_features_batch.to(device=device)
                            # target_captions_stack = target_captions_stack.to(device=device)
                            target_idx_batch = target_idx_batch.to(device=device)
                            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch,
                                                                                    num_distractors + 1)  # before squeezing!
                            target_idx_batch = torch.squeeze(target_idx_batch)

                            seq, log_p = sender_inner(target_encoded_features, seq_data=None, device=device)
                            seq = seq.detach()  # technically not necessary but kind of nicer

                            logits = receiver_inner(all_features_batch, seq)
                            loss = criterion(logits, target_idx_batch)
                            commentaries = commentary_network(sender_idx_batch[sender_idx], receiver_idx_batch[receiver_idx])

                            #print(loss.size(), commentaries.size())

                            # Receiver update (classic supervised learning)
                            receiver_loss = torch.mean(loss*commentaries)
                            #this is higher syntax for opt.zero_grad() loss.backward() opt.step()
                            optimizer_receiver_inner.step(receiver_loss)

                            # Sender update REINFORCE!
                            log_p_mask = prob_mask(seq)
                            log_p = log_p * log_p_mask
                            log_p = torch.sum(log_p, dim=1)
                            value = -loss.detach()
                            baselined_value = value - baselines[sender_idx, receiver_idx]
                            sender_reinforce_objective = log_p * baselined_value
                            #print(sender_reinforce_objective.size(), commentaries.size())
                            sender_loss = torch.mean(-sender_reinforce_objective*commentaries)
                            #this is higher syntax for opt.zero_grad() loss.backward() opt.step()
                            optimizer_sender_inner.step(sender_loss)

                            avg_value = torch.mean(value)
                            baselines[sender_idx, receiver_idx] = baseline_polyak * baselines[sender_idx, receiver_idx] + (
                                        1. - baseline_polyak) * avg_value

                            training_loss_grid[sender_idx][receiver_idx].update(-torch.mean(value).to(device='cpu'))
                            training_acc_grid[sender_idx][receiver_idx].update(logits.to(device='cpu'),
                                                                               target_idx_batch.to(device='cpu'))
                            entropies[sender_idx].update(-torch.mean(log_p.to(device='cpu')))
                            commentary_weight_grid[sender_idx][receiver_idx].update(torch.mean(commentaries).to('cpu'))



                    # validation for commentary loss
                    all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch = val_data
                    all_features_batch = all_features_batch.to(device=device)
                    target_captions_stack = target_captions_stack.to(device=device)
                    target_idx_batch = target_idx_batch.to(device=device)

                    #validation for sender
                    target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch, num_distractors+1)
                    logits = sender_inner(target_encoded_features, target_captions_stack[:, :-1], device=device)
                    val_loss_sender = criterion(torch.swapaxes(logits, 1,2), target_captions_stack[:,1:])
                    #print(loss.size(), prob_mask(prob_mask(target_captions_stack))[:,1:].size())
                    #print(prob_mask(target_captions_stack)[:,1:].cpu().detach())
                    val_loss_sender = val_loss_sender*prob_mask(target_captions_stack)[:,1:]
                    val_loss_sender = torch.mean(val_loss_sender)

                    #validation for receiver
                    target_idx_batch = torch.squeeze(target_idx_batch)
                    logits = receiver_inner(all_features_batch, target_captions_stack)
                    val_loss_receiver = criterion(logits, target_idx_batch)
                    val_loss_receiver = torch.mean(val_loss_receiver)

                    #print('val losses sender, receiver', val_loss_sender, val_loss_receiver)
                    val_loss = val_loss_sender + val_loss_receiver
                    val_loss.backward()
                    optimizer_commentary_network.step()



        #test
        sender_write_flags = [True for _ in senders]

        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in utils.repeat_dataset(data_loader_test, repeats_per_epoch):

            receiver_idx = random.randrange(0, num_receivers)
            sender_idx = random.randrange(0, num_senders)
            sender = senders[sender_idx]
            receiver = receivers[receiver_idx]
            receiver.eval()
            sender.eval()

            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch,
                                                                    num_distractors + 1)  # before squeezing!
            target_idx_batch = torch.squeeze(target_idx_batch)

            seq, log_p = sender(target_encoded_features, seq_data=None, device=device)
            seq = seq.detach()  # technically not necessary but kind of nicer

            logits = receiver(all_features_batch, seq)
            loss = criterion(logits, target_idx_batch)

            value = -loss.detach()
            test_loss_grid[sender_idx][receiver_idx].update(-torch.mean(value).to(device='cpu'))
            test_acc_grid[sender_idx][receiver_idx].update(logits.to(device='cpu'), target_idx_batch.to(device='cpu'))

            supervised_sender_test(sender, criterion, test_loss=sup_sender_loss[sender_idx], test_pp=sup_sender_pp[sender_idx], target_encoded_features=target_encoded_features, target_captions_stack=target_captions_stack, vocab=vocab, write_flag=sender_write_flags[sender_idx], writer=writer, sentence_tag=f'sender_{sender_idx}', episode=episode, device=device)
            supervised_receiver_test(receiver, criterion, test_loss=sup_receiver_loss[receiver_idx], test_acc=sup_receiver_acc[receiver_idx], target_idx_batch=target_idx_batch, all_features_batch=all_features_batch, target_captions_stack=target_captions_stack)
            sender_write_flags[sender_idx] = False
        # end of episode stuff
        [scheduler_receiver.step() for scheduler_receiver in schedulers_receiver]
        [scheduler_sender.step() for scheduler_sender in schedulers_sender]

        entropy_results = []
        training_acc_result = []
        training_loss_result = []
        test_acc_result = []
        test_loss_result = []
        commentary_weights = []

        write_dict = {}
        for sender_idx in range(num_senders):
            for receiver_idx in range(num_receivers):
                training_loss = training_loss_grid[sender_idx][receiver_idx]
                episode_training_loss = training_loss.compute()
                training_loss.reset()

                training_acc = training_acc_grid[sender_idx][receiver_idx]
                episode_training_acc = utils.safe_compute_accuracy_metric(training_acc)
                training_acc.reset()

                test_loss = test_loss_grid[sender_idx][receiver_idx]
                episode_test_loss = test_loss.compute()
                test_loss.reset()

                test_acc = test_acc_grid[sender_idx][receiver_idx]
                episode_test_acc = utils.safe_compute_accuracy_metric(test_acc)
                test_acc.reset()

                commentary_weight = commentary_weight_grid[sender_idx][receiver_idx]
                episode_commentary_weight = commentary_weight.compute()
                commentary_weight.reset()

                write_dict[f'training_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_loss
                write_dict[f'training_acc_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_acc
                write_dict[f'test_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_test_loss
                write_dict[f'training_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_loss
                write_dict[f'commentary_weights_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_commentary_weight

                training_acc_result.append(episode_training_acc)
                training_loss_result.append(episode_training_loss)
                test_acc_result.append(episode_test_acc)
                test_loss_result.append(episode_test_loss)

        for sender_idx in range(num_senders):
            episode_entropy = entropies[sender_idx].compute()
            entropies[sender_idx].reset()
            write_dict[f'entropy_sender_{sender_idx}'] = episode_entropy
            entropy_results.append(episode_entropy)

            episode_sup_sender_pp = sup_sender_pp[sender_idx].compute()
            sup_sender_pp[sender_idx].reset()
            write_dict[f'supervised_pp_sender_{sender_idx}'] = episode_sup_sender_pp

            episode_sup_sender_loss = sup_sender_loss[sender_idx].compute()
            sup_sender_loss[sender_idx].reset()
            write_dict[f'supervised_loss_sender_{sender_idx}'] = episode_sup_sender_loss

        for receiver_idx in range(num_receivers):
            episode_sup_receiver_acc = utils.safe_compute_accuracy_metric(sup_receiver_acc[receiver_idx])
            sup_receiver_acc[receiver_idx].reset()
            write_dict[f'supervised_acc_receiver_{receiver_idx}'] = episode_sup_receiver_acc

            episode_sup_receiver_loss = sup_receiver_loss[receiver_idx].compute()
            sup_receiver_loss[receiver_idx].reset()
            write_dict[f'supervised_loss_receiver_[receiver_idx'] = episode_sup_receiver_loss



        p_bar.set_description(
            f'Train: L{np.mean(training_loss_result) :.3e} / ACC{np.mean(training_acc_result) :.3e} || Test: L{np.mean(test_loss_result) :.3e} / ACC{np.mean(test_acc_result) :.3e}')
        writer.add_scalars(main_tag=writer_tag,
                           tag_scalar_dict=write_dict,
                           global_step=episode)

    writer.flush()

def weighted_softmax_commentary_training_interactive_only(senders, receivers, commentary_network, receiver_lr, sender_lr, commentary_lr, num_distractors, path, writer_tag, num_inner_loop_steps=2, num_episodes=200, batch_size=512, num_workers=4, repeats_per_epoch=1, device='cpu', baseline_polyak=0.99):
    # indexing here will generally be first sender index, then receiver index for anything that has num_senders x num_receivers elements arranged in a 2d grid

    num_senders = len(senders)
    num_receivers = len(receivers)

    [sender.to(device) for sender in senders]
    [receiver.to(device) for receiver in receivers]
    commentary_network.to(device)

    [receiver.to(device) for receiver in receivers]
    [receiver.train() for receiver in receivers]
    [sender.to(device) for sender in senders]
    [sender.train() for sender in senders]

    optimizers_receiver = [torch.optim.Adam(receiver.parameters(), lr=receiver_lr) for receiver in receivers]
    optimizers_sender = [torch.optim.Adam(sender.parameters(), lr=sender_lr) for sender in senders]
    optimizer_commentary_network = torch.optim.Adam(commentary_network.parameters(), lr=commentary_lr)

    schedulers_receiver = [torch.optim.lr_scheduler.StepLR(optimizer_receiver, 1.0, gamma=0.95) for optimizer_receiver
                           in optimizers_receiver]
    schedulers_sender = [torch.optim.lr_scheduler.StepLR(optimizer_sender, 1.0, gamma=0.95) for optimizer_sender in
                         optimizers_sender]

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    baselines = torch.zeros(size=(num_senders, num_receivers)).to(device=device)

    sender_idx_batch = []
    receiver_idx_batch = []
    idx_tracker = []
    running_idx = 0
    for sender_idx in range(num_senders):
        idx_tracker.append([])
        for receiver_idx in range(num_receivers):
            idx_tracker[sender_idx].append(running_idx)
            sender_idx_batch.append(torch.ones((batch_size,1), dtype=int)*sender_idx)
            receiver_idx_batch.append(torch.ones((batch_size,1), dtype=int)*receiver_idx)
            running_idx = running_idx +1
    sender_idx_batch = torch.cat(sender_idx_batch, dim=-1).to(device=device)
    receiver_idx_batch = torch.cat(receiver_idx_batch, dim=-1).to(device=device)
    print(sender_idx_batch.dtype)


    (train_ds, test_ds), vocab = data.load_prepared_coco_data()

    data_loaders_train = [data.create_data_loader(
        train_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    ) for _ in range(num_inner_loop_steps)]

    # split test data in val and test data
    test_val_cutoff = int(len(test_ds) * TEST_VAL_SPLIT)
    validation_ds = test_ds[test_val_cutoff:]
    test_ds = test_ds[:test_val_cutoff]
    data_loader_test = data.create_data_loader(
        test_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    data_loader_validation = data.create_data_loader(
        validation_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    # metrics
    writer = SummaryWriter(path)
    training_loss_grid = [[torchmetrics.MeanMetric() for _ in range(num_receivers)] for _ in range(num_senders)]
    training_acc_grid = [[torchmetrics.Accuracy() for _ in range(num_receivers)] for _ in range(num_senders)]
    test_loss_grid = [[torchmetrics.MeanMetric() for _ in range(num_receivers)] for _ in range(num_senders)]
    test_acc_grid = [[torchmetrics.Accuracy() for _ in range(num_receivers)] for _ in range(num_senders)]
    commentary_weight_grid =[[torchmetrics.MeanMetric() for _ in range(num_receivers)] for _ in range(num_senders)]

    sup_sender_loss = [torchmetrics.MeanMetric() for _ in range(num_senders)]
    sup_sender_pp = [torchmetrics.MeanMetric() for _ in range(num_senders)]
    sup_receiver_loss = [torchmetrics.MeanMetric() for _ in range(num_receivers)]
    sup_receiver_acc = [torchmetrics.Accuracy() for _ in range(num_receivers)]

    #validation is not interactive testing but
    entropies = [torchmetrics.MeanMetric() for _ in range(num_senders)]

    for episode in (p_bar := tqdm(range(num_episodes))):
        zipped_training_data_loader = zip(*data_loaders_train)
        training_val_data_loader = zip(utils.cycle_dataloader(data_loader_validation), zipped_training_data_loader)
        for val_data, inner_loop_step_data in utils.repeat_dataset(training_val_data_loader, repeats_per_epoch):

            optimizer_commentary_network.zero_grad()
            receiver_idx = random.randrange(0, num_receivers)
            sender_idx = random.randrange(0, num_senders)
            sender = senders[sender_idx]
            receiver = receivers[receiver_idx]
            receiver.train()
            sender.train()

            optimizer_sender = optimizers_sender[sender_idx]
            optimizer_receiver = optimizers_receiver[receiver_idx]
            optimizer_receiver.zero_grad()
            optimizer_sender.zero_grad()

            #inner loop
            with higher.innerloop_ctx(model=sender, opt=optimizer_sender, copy_initial_weights=False) as (
            sender_inner, optimizer_sender_inner):
                with higher.innerloop_ctx(model=receiver, opt=optimizer_receiver, copy_initial_weights=False) as (
                receiver_inner, optimizer_receiver_inner):
                    for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in inner_loop_step_data:

                        #cudnn does not support double backwards pass :(
                        with torch.backends.cudnn.flags(enabled=False):

                            all_features_batch = all_features_batch.to(device=device)
                            # target_captions_stack = target_captions_stack.to(device=device)
                            target_idx_batch = target_idx_batch.to(device=device)
                            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch,
                                                                                    num_distractors + 1)  # before squeezing!
                            target_idx_batch = torch.squeeze(target_idx_batch)

                            seq, log_p = sender_inner(target_encoded_features, seq_data=None, device=device)
                            seq = seq.detach()  # technically not necessary but kind of nicer

                            logits = receiver_inner(all_features_batch, seq)
                            loss = criterion(logits, target_idx_batch)
                            commentaries = commentary_network(target_encoded_features, sender_idx_batch, receiver_idx_batch)[:,idx_tracker[sender_idx][receiver_idx]]

                            #print(loss.size(), commentaries.size())

                            # Receiver update (classic supervised learning)
                            receiver_loss = torch.mean(loss*commentaries)
                            #this is higher syntax for opt.zero_grad() loss.backward() opt.step()
                            optimizer_receiver_inner.step(receiver_loss)

                            # Sender update REINFORCE!
                            log_p_mask = prob_mask(seq)
                            log_p = log_p * log_p_mask
                            log_p = torch.sum(log_p, dim=1)
                            value = -loss.detach()
                            baselined_value = value - baselines[sender_idx, receiver_idx]
                            sender_reinforce_objective = log_p * baselined_value
                            #print(sender_reinforce_objective.size(), commentaries.size())
                            sender_loss = torch.mean(-sender_reinforce_objective*commentaries)
                            #this is higher syntax for opt.zero_grad() loss.backward() opt.step()
                            optimizer_sender_inner.step(sender_loss)

                            avg_value = torch.mean(value)
                            baselines[sender_idx, receiver_idx] = baseline_polyak * baselines[sender_idx, receiver_idx] + (
                                        1. - baseline_polyak) * avg_value

                            training_loss_grid[sender_idx][receiver_idx].update(-torch.mean(value).to(device='cpu'))
                            training_acc_grid[sender_idx][receiver_idx].update(logits.to(device='cpu'),
                                                                               target_idx_batch.to(device='cpu'))
                            entropies[sender_idx].update(-torch.mean(log_p.to(device='cpu')))
                            commentary_weight_grid[sender_idx][receiver_idx].update(torch.mean(commentaries).to('cpu'))



                    # validation for commentary loss
                    all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch = val_data
                    all_features_batch = all_features_batch.to(device=device)
                    target_captions_stack = target_captions_stack.to(device=device)
                    target_idx_batch = target_idx_batch.to(device=device)

                    #validation for sender
                    target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch, num_distractors+1)
                    logits = sender_inner(target_encoded_features, target_captions_stack[:, :-1], device=device)
                    val_loss_sender = criterion(torch.swapaxes(logits, 1,2), target_captions_stack[:,1:])
                    #print(loss.size(), prob_mask(prob_mask(target_captions_stack))[:,1:].size())
                    #print(prob_mask(target_captions_stack)[:,1:].cpu().detach())
                    val_loss_sender = val_loss_sender*prob_mask(target_captions_stack)[:,1:]
                    val_loss_sender = torch.mean(val_loss_sender)

                    #validation for receiver
                    target_idx_batch = torch.squeeze(target_idx_batch)
                    logits = receiver_inner(all_features_batch, target_captions_stack)
                    val_loss_receiver = criterion(logits, target_idx_batch)
                    val_loss_receiver = torch.mean(val_loss_receiver)

                    #print('val losses sender, receiver', val_loss_sender, val_loss_receiver)
                    val_loss = val_loss_sender + val_loss_receiver
                    val_loss.backward()
                    optimizer_commentary_network.step()



        #test
        sender_write_flags = [True for _ in senders]

        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in utils.repeat_dataset(data_loader_test, repeats_per_epoch):

            receiver_idx = random.randrange(0, num_receivers)
            sender_idx = random.randrange(0, num_senders)
            sender = senders[sender_idx]
            receiver = receivers[receiver_idx]
            receiver.eval()
            sender.eval()

            all_features_batch = all_features_batch.to(device=device)
            target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch,
                                                                    num_distractors + 1)  # before squeezing!
            target_idx_batch = torch.squeeze(target_idx_batch)

            seq, log_p = sender(target_encoded_features, seq_data=None, device=device)
            seq = seq.detach()  # technically not necessary but kind of nicer

            logits = receiver(all_features_batch, seq)
            loss = criterion(logits, target_idx_batch)

            value = -loss.detach()
            test_loss_grid[sender_idx][receiver_idx].update(-torch.mean(value).to(device='cpu'))
            test_acc_grid[sender_idx][receiver_idx].update(logits.to(device='cpu'), target_idx_batch.to(device='cpu'))

            supervised_sender_test(sender, criterion, test_loss=sup_sender_loss[sender_idx], test_pp=sup_sender_pp[sender_idx], target_encoded_features=target_encoded_features, target_captions_stack=target_captions_stack, vocab=vocab, write_flag=sender_write_flags[sender_idx], writer=writer, sentence_tag=f'sender_{sender_idx}', episode=episode)
            supervised_receiver_test(receiver, criterion, test_loss=sup_receiver_loss[receiver_idx], test_acc=sup_receiver_acc[receiver_idx], target_idx_batch=target_idx_batch, all_features_batch=all_features_batch, target_captions_stack=target_captions_stack)
            sender_write_flags[sender_idx] = False

        # end of episode stuff
        [scheduler_receiver.step() for scheduler_receiver in schedulers_receiver]
        [scheduler_sender.step() for scheduler_sender in schedulers_sender]

        entropy_results = []
        training_acc_result = []
        training_loss_result = []
        test_acc_result = []
        test_loss_result = []
        commentary_weights = []

        write_dict = {}
        for sender_idx in range(num_senders):
            for receiver_idx in range(num_receivers):
                training_loss = training_loss_grid[sender_idx][receiver_idx]
                episode_training_loss = training_loss.compute()
                training_loss.reset()

                training_acc = training_acc_grid[sender_idx][receiver_idx]
                episode_training_acc = utils.safe_compute_accuracy_metric(training_acc)
                training_acc.reset()

                test_loss = test_loss_grid[sender_idx][receiver_idx]
                episode_test_loss = test_loss.compute()
                test_loss.reset()

                test_acc = test_acc_grid[sender_idx][receiver_idx]
                episode_test_acc = utils.safe_compute_accuracy_metric(test_acc)
                test_acc.reset()

                commentary_weight = commentary_weight_grid[sender_idx][receiver_idx]
                episode_commentary_weight = commentary_weight.compute()
                commentary_weight.reset()

                write_dict[f'training_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_loss
                write_dict[f'training_acc_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_acc
                write_dict[f'test_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_test_loss
                write_dict[f'training_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_loss
                write_dict[f'commentary_weights_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_commentary_weight

                training_acc_result.append(episode_training_acc)
                training_loss_result.append(episode_training_loss)
                test_acc_result.append(episode_test_acc)
                test_loss_result.append(episode_test_loss)

        for sender_idx in range(num_senders):
            episode_entropy = entropies[sender_idx].compute()
            entropies[sender_idx].reset()
            write_dict[f'entropy_sender_{sender_idx}'] = episode_entropy
            entropy_results.append(episode_entropy)

            episode_sup_sender_pp = sup_sender_pp[sender_idx].compute()
            sup_sender_pp[sender_idx].reset()
            write_dict[f'supervised_pp_sender_{sender_idx}'] = episode_sup_sender_pp

            episode_sup_sender_loss = sup_sender_loss[sender_idx].compute()
            sup_sender_loss[sender_idx].reset()
            write_dict[f'supervised_loss_sender_{sender_idx}'] = episode_sup_sender_loss

        for receiver_idx in range(num_receivers):
            episode_sup_receiver_acc = utils.safe_compute_accuracy_metric(sup_receiver_acc[receiver_idx])
            sup_receiver_acc[receiver_idx].reset()
            write_dict[f'supervised_acc_receiver_{receiver_idx}'] = episode_sup_receiver_acc

            episode_sup_receiver_loss = sup_receiver_loss[receiver_idx].compute()
            sup_receiver_loss[receiver_idx].reset()
            write_dict[f'supervised_loss_receiver_[receiver_idx'] = episode_sup_receiver_loss



        p_bar.set_description(
            f'Train: L{np.mean(training_loss_result) :.3e} / ACC{np.mean(training_acc_result) :.3e} || Test: L{np.mean(test_loss_result) :.3e} / ACC{np.mean(test_acc_result) :.3e}')
        writer.add_scalars(main_tag=writer_tag,
                           tag_scalar_dict=write_dict,
                           global_step=episode)

    writer.flush()

def tscl_multiagent_training_interactive_only(senders, receivers, receiver_lr, sender_lr, num_distractors, path, writer_tag, epsilon=0.1, fifo_size=10, num_episodes=200, batch_size=512, num_workers=4, repeats_per_epoch=1, device='cpu', baseline_polyak=0.99):

    #indexing here will generally be first sender index, then receiver index for anything that has num_senders x num_receivers elements arranged in a 2d grid

    num_senders = len(senders)
    num_receivers = len(receivers)

    [receiver.to(device) for receiver in receivers]
    [receiver.train() for receiver in receivers]
    [sender.to(device) for sender in senders]
    [sender.train() for sender in senders]

    optimizers_receiver = [torch.optim.Adam(receiver.parameters(), lr=receiver_lr) for receiver in receivers]
    optimizers_sender = [torch.optim.Adam(sender.parameters(), lr=sender_lr) for sender in senders]

    schedulers_receiver = [torch.optim.lr_scheduler.StepLR(optimizer_receiver, 1.0, gamma=0.95) for optimizer_receiver in optimizers_receiver]
    schedulers_sender = [torch.optim.lr_scheduler.StepLR(optimizer_sender, 1.0, gamma=0.95) for optimizer_sender in optimizers_sender]

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    baselines = torch.zeros(size=(num_senders,num_receivers)).to(device=device)

    (train_ds, test_ds), vocab = data.load_prepared_coco_data()

    data_loader_train = data.create_data_loader(
        train_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )

    #split test data in val and test data
    test_val_cutoff = int(len(train_ds)*TEST_VAL_SPLIT)
    validation_ds = test_ds[test_val_cutoff:]
    test_ds = test_ds[:test_val_cutoff]
    data_loader_test = data.create_data_loader(
        test_ds, batch_size=batch_size, num_distractors=num_distractors, num_workers=num_workers, device=device
    )
    #metrics
    writer = SummaryWriter(path)
    training_loss_grid = [[torchmetrics.MeanMetric() for _ in range(num_receivers)] for _ in range(num_senders)]
    training_acc_grid = [[torchmetrics.Accuracy() for _ in range(num_receivers)] for _ in range(num_senders)]
    test_loss_grid = [[torchmetrics.MeanMetric() for _ in range(num_receivers)] for _ in range(num_senders)]
    test_acc_grid = [[torchmetrics.Accuracy() for _ in range(num_receivers)] for _ in range(num_senders)]
    entropies = [torchmetrics.MeanMetric() for _ in range(num_senders)]

    tscl = utils.tscl_helper(num_senders, num_receivers, fifo_size=fifo_size)

    for episode in (p_bar := tqdm(range(num_episodes))):

        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in utils.repeat_dataset(data_loader_train, repeats_per_epoch):
            sender_idx, receiver_idx = tscl.sample_epsilon_greedy(epsilon=epsilon)
            sender = senders[sender_idx]
            receiver = receivers[receiver_idx]
            receiver.train()
            sender.train()

            optimizer_sender = optimizers_sender[sender_idx]
            optimizer_receiver = optimizers_receiver[receiver_idx]

            optimizer_receiver.zero_grad()
            optimizer_sender.zero_grad()

            all_features_batch = all_features_batch.to(device=device)
            #target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch, num_distractors+1)#before squeezing!
            target_idx_batch = torch.squeeze(target_idx_batch)


            seq, log_p = sender(target_encoded_features, seq_data=None, device=device)
            seq = seq.detach() # technically not necessary but kind of nicer

            logits = receiver(all_features_batch, seq)
            loss = criterion(logits, target_idx_batch)

            #Receiver update (classic supervised learning)
            receiver_loss = torch.mean(loss)
            receiver_loss.backward()
            optimizer_receiver.step()

            #Sender update REINFORCE!
            log_p_mask = prob_mask(seq)
            log_p = log_p*log_p_mask
            log_p = torch.sum(log_p, dim=1)
            value = -loss.detach()
            baselined_value = value - baselines[sender_idx, receiver_idx]
            sender_reinforce_objective = log_p*baselined_value
            sender_loss = torch.mean(-sender_reinforce_objective)
            sender_loss.backward()
            optimizer_sender.step()

            avg_value = torch.mean(value)
            baselines[sender_idx, receiver_idx] = baseline_polyak*baselines[sender_idx, receiver_idx] + (1.-baseline_polyak)*avg_value

            training_loss_grid[sender_idx][receiver_idx].update(-torch.mean(value).to(device='cpu'))
            training_acc_grid[sender_idx][receiver_idx].update(logits.to(device='cpu'), target_idx_batch.to(device='cpu'))
            entropies[sender_idx].update(-torch.mean(log_p.to(device='cpu')))
            tscl.update(sender_idx, receiver_idx, -torch.mean(value).to(device='cpu').numpy())



        for all_features_batch, target_features_batch, target_captions_stack, target_idx_batch, ids_batch in utils.repeat_dataset(data_loader_test, repeats_per_epoch):

            receiver_idx = random.randrange(0, num_receivers)
            sender_idx = random.randrange(0, num_senders)
            sender = senders[sender_idx]
            receiver = receivers[receiver_idx]
            receiver.eval()
            sender.eval()

            all_features_batch = all_features_batch.to(device=device)
            #target_captions_stack = target_captions_stack.to(device=device)
            target_idx_batch = target_idx_batch.to(device=device)
            target_encoded_features = target_distractor_encode_data(all_features_batch, target_idx_batch, num_distractors+1)#before squeezing!
            target_idx_batch = torch.squeeze(target_idx_batch)


            seq, log_p = sender(target_encoded_features, seq_data=None, device=device)
            seq = seq.detach() # technically not necessary but kind of nicer

            logits = receiver(all_features_batch, seq)
            loss = criterion(logits, target_idx_batch)

            value = -loss.detach()
            test_loss_grid[sender_idx][receiver_idx].update(-torch.mean(value).to(device='cpu'))
            test_acc_grid[sender_idx][receiver_idx].update(logits.to(device='cpu'), target_idx_batch.to(device='cpu'))

        #end of episode stuff
        [scheduler_receiver.step() for scheduler_receiver in schedulers_receiver]
        [scheduler_sender.step() for scheduler_sender in schedulers_sender]

        entropy_results = []
        training_acc_result = []
        training_loss_result = []
        test_acc_result = []
        test_loss_result = []

        write_dict = {}
        for sender_idx in range(num_senders):
            episode_entropy = entropies[sender_idx].compute()
            entropies[sender_idx].reset()
            write_dict[f'entropy_sender_{sender_idx}'] = episode_entropy
            entropy_results.append(episode_entropy)
            for receiver_idx in range(num_receivers):
                training_loss = training_loss_grid[sender_idx][receiver_idx]
                episode_training_loss = training_loss.compute()
                training_loss.reset()

                training_acc = training_acc_grid[sender_idx][receiver_idx]
                episode_training_acc = utils.safe_compute_accuracy_metric(training_acc)
                training_acc.reset()

                test_loss = test_loss_grid[sender_idx][receiver_idx]
                episode_test_loss = test_loss.compute()
                test_loss.reset()

                test_acc = test_acc_grid[sender_idx][receiver_idx]
                episode_test_acc = utils.safe_compute_accuracy_metric(test_acc)
                test_acc.reset()

                #tscl value of each task
                flat_idx = tscl.flat_idx(sender_idx, receiver_idx)
                tscl_val = tscl.task_rewards[flat_idx]

                write_dict[f'tscl_sender_{sender_idx}_receiver_{receiver_idx}'] = tscl_val
                write_dict[f'training_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_loss
                write_dict[f'training_acc_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_acc
                write_dict[f'test_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_test_loss
                write_dict[f'training_loss_sender_{sender_idx}_receiver_{receiver_idx}'] = episode_training_loss

                training_acc_result.append(episode_training_acc)
                training_loss_result.append(episode_training_loss)
                test_acc_result.append(episode_test_acc)
                test_loss_result.append(episode_test_loss)

        p_bar.set_description(
            f'Train: L{np.mean(training_loss_result) :.3e} / ACC{np.mean(training_acc_result) :.3e} || Test: L{np.mean(test_loss_result) :.3e} / ACC{np.mean(test_acc_result) :.3e}')
        writer.add_scalars(main_tag=writer_tag,
                           tag_scalar_dict=write_dict,
                           global_step=episode)

    writer.flush()

if __name__ == '__main__':

    path = f'results/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}/'
    writer_tag = 'idx:commentaries'
    num_distractors = 1
    pretrain_episodes = 0

    device = torch.device('cuda:0')
    num_senders = 2
    num_receivers = 2

    receiver_lr = 0.000001
    sender_lr = 0.000001

    senders = [agents.lstm_sender_agent(feature_size=2049, text_embedding_size=128, vocab_size=2000, lstm_size=128,
                                        lstm_depth=2, feature_embedding_hidden_size=64) for _ in range(num_senders)]
    pretrain_sender = lambda sender: training.pretrain_sender_lstm(sender=sender, path=path + 'sender_pretraining',
                                                                   writer_tag='a_sender', batch_size=256,
                                                                   num_distractors=num_distractors,
                                                                   num_episodes=pretrain_episodes, device=device)
    senders = [pretrain_sender(sender) for sender in tqdm(senders)]

    receivers = [agents.lstm_receiver_agent(feature_size=2048, text_embedding_size=128, vocab_size=2000, lstm_size=128,
                                            lstm_depth=2, feature_embedding_hidden_size=64, readout_hidden_size=32) for
                 _ in range(num_receivers)]

    pretrain_receiver = lambda receiver: training.pretrain_receiver_lstm(receiver=receiver,
                                                                         num_episodes=pretrain_episodes,
                                                                         path=path + 'receiver_pretraining',
                                                                         writer_tag='a_l_receiver',
                                                                         batch_size=128,
                                                                         num_distractors=num_distractors, lr=0.0001,
                                                                         device=device)
    receivers = [pretrain_receiver(receiver) for receiver in tqdm(receivers)]
    tscl_multiagent_training_interactive_only(senders, receivers, receiver_lr, sender_lr, num_distractors, path, writer_tag, fifo_size=10, device=device)


    """
    path = f'results/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}/'
    writer_tag = 'idx:commentaries'
    num_distractors = 1
    pretrain_episodes = 0

    device = torch.device('cuda:0')
    num_senders = 2
    num_receivers = 2

    receiver_lr = 0.000001
    sender_lr = 0.000001
    commentary_lr = 0.0001

    commentary_nn = commentary_networks.objects_commentary_network_normalized(num_senders, num_receivers, 64,2049,2,2,64,4)
    senders = [agents.lstm_sender_agent(feature_size=2049, text_embedding_size=128, vocab_size=2000, lstm_size=128, lstm_depth=2, feature_embedding_hidden_size=64) for _ in range(num_senders)]
    pretrain_sender = lambda sender: training.pretrain_sender_lstm(sender=sender, path=path + 'sender_pretraining',
                                  writer_tag='a_sender', batch_size=256, num_distractors=num_distractors,
                                  num_episodes=pretrain_episodes, device=device)
    senders = [pretrain_sender(sender) for sender in tqdm(senders)]

    receivers = [agents.lstm_receiver_agent(feature_size=2048, text_embedding_size=128, vocab_size=2000, lstm_size=128, lstm_depth=2, feature_embedding_hidden_size=64, readout_hidden_size=32) for _ in range(num_receivers)]

    pretrain_receiver = lambda receiver: training.pretrain_receiver_lstm(receiver=receiver, num_episodes=pretrain_episodes,
                                      path=path + 'receiver_pretraining', writer_tag='a_l_receiver',
                                      batch_size=128, num_distractors=num_distractors, lr=0.0001, device=device)
    receivers = [pretrain_receiver(receiver) for receiver in tqdm(receivers)]
    weighted_softmax_commentary_training_interactive_only(senders, receivers, commentary_nn, receiver_lr, sender_lr, commentary_lr, num_distractors, path, writer_tag, batch_size=256, device=device)

    """

    """
    path = f'results/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}/'
    writer_tag = 'idx:commentaries'
    num_distractors = 1
    pretrain_episodes = 0

    device = torch.device('cuda:0')
    num_senders = 2
    num_receivers = 2

    receiver_lr = 0.000001
    sender_lr = 0.000001
    commentary_lr = 0.0001

    commentary_nn = commentary_networks.idx_commentary_network(num_senders, num_receivers, 16,16)
    senders = [agents.lstm_sender_agent(feature_size=2049, text_embedding_size=128, vocab_size=2000, lstm_size=128, lstm_depth=2, feature_embedding_hidden_size=64) for _ in range(num_senders)]
    pretrain_sender = lambda sender: training.pretrain_sender_lstm(sender=sender, path=path + 'sender_pretraining',
                                  writer_tag='a_sender', batch_size=256, num_distractors=num_distractors,
                                  num_episodes=pretrain_episodes, device=device)
    senders = [pretrain_sender(sender) for sender in tqdm(senders)]

    receivers = [agents.lstm_receiver_agent(feature_size=2048, text_embedding_size=128, vocab_size=2000, lstm_size=128, lstm_depth=2, feature_embedding_hidden_size=64, readout_hidden_size=32) for _ in range(num_receivers)]

    pretrain_receiver = lambda receiver: training.pretrain_receiver_lstm(receiver=receiver, num_episodes=pretrain_episodes,
                                      path=path + 'receiver_pretraining', writer_tag='a_l_receiver',
                                      batch_size=128, num_distractors=num_distractors, lr=0.0001, device=device)
    receivers = [pretrain_receiver(receiver) for receiver in tqdm(receivers)]
    idx_commentary_training_interactive_only(senders, receivers, commentary_nn, receiver_lr, sender_lr, commentary_lr, num_distractors, path, writer_tag, batch_size=256, device=device)
    """