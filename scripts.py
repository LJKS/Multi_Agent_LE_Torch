import training
import marl_training
import agents
import commentary_networks
import data
import argparse
import os
import torch
from datetime import datetime
from tqdm import tqdm



def summarize_key_args(args, key_args):
    summary=''
    args=vars(args)
    for key in key_args:
        summary = summary+'--'+key+'_'+str(args[key])
    return summary


def tscl_population_training_lstm128(args):
    key_args = ['num_senders', 'num_receivers', 'num_distractors', 'fifo_size', 'epsilon']
    path = f'results/{args.experiment}/{summarize_key_args(args, key_args)}/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}/'
    path = os.getcwd() + '/' path

    save_path = path + 'saves/'
    for which_training in ['pretraining', 'finetuning']:
        for agent in ['sender', 'receiver']:
            sub_dir = which_training + '_' + agent
            os.makedirs(save_path + sub_dir)

    writer_tag = args.tag
    num_distractors = args.num_distractors

    device = torch.device('cuda:0')
    num_senders = args.num_senders
    num_receivers = args.num_receivers

    pretraining_lr = args.pretraining_lr
    receiver_lr = args.finetuning_lr
    sender_lr = args.finetuning_lr

    batch_size = args.batch_size

    pretraining_epochs = args.pretraining_epochs
    finetuning_epochs = args.finetuning_epochs

    epsilon = args.epsilon
    fifo_size = args.fifo_size

    senders = [agents.lstm_sender_agent(feature_size=2049, text_embedding_size=128, vocab_size=2000, lstm_size=128,
                                        lstm_depth=2, feature_embedding_hidden_size=64) for _ in range(num_senders)]
    pretrain_sender = lambda sender: training.pretrain_sender_lstm(sender=sender, path=path + 'sender_pretraining',
                                                                   writer_tag=writer_tag, batch_size=batch_size,
                                                                   num_distractors=num_distractors,
                                                                   num_episodes=pretraining_epochs, lr=pretraining_lr,
                                                                   device=device)
    print('Pretraining senders:')
    senders = [pretrain_sender(sender) for sender in tqdm(senders)]

    receivers = [agents.lstm_receiver_agent(feature_size=2048, text_embedding_size=128, vocab_size=2000, lstm_size=128,
                                            lstm_depth=2, feature_embedding_hidden_size=64, readout_hidden_size=32) for
                 _ in range(num_receivers)]

    pretrain_receiver = lambda receiver: training.pretrain_receiver_lstm(receiver=receiver,
                                                                         num_episodes=pretraining_epochs,
                                                                         path=path + 'receiver_pretraining',
                                                                         writer_tag=writer_tag,
                                                                         batch_size=batch_size,
                                                                         num_distractors=num_distractors,
                                                                         lr=pretraining_lr, device=device)
    print('Pretraining receivers:')
    receivers = [pretrain_receiver(receiver) for receiver in tqdm(receivers)]
    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'pretraining' + '_' + what_agent + '/'
            file_name = what_agent + '_' + str(num) + '.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)

    print('Interactive finetuning')
    marl_training.tscl_multiagent_training_interactive_only(senders=senders, receivers=receivers, receiver_lr=receiver_lr, sender_lr=sender_lr, num_distractors=num_distractors, path=path+'finetuning', writer_tag=writer_tag, fifo_size=fifo_size, num_episodes=finetuning_epochs, batch_size=batch_size, epsilon=epsilon, device=device)

    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'finetuning' + '_' + what_agent + '/'
            file_name = what_agent + '_' + str(num) + '.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)

def commentary_idx_training_lstm128(args):
    key_args = ['num_senders', 'num_receivers', 'num_distractors']
    path = f'results/{args.experiment}/{summarize_key_args(args, key_args)}/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}/'

    save_path = path + 'saves/'
    for which_training in ['pretraining', 'finetuning']:
        for agent in ['sender', 'receiver']:
            sub_dir = which_training + '_' + agent
            os.makedirs(save_path + sub_dir)

    writer_tag = args.tag
    num_distractors = args.num_distractors

    device = torch.device('cuda:0')
    num_senders = args.num_senders
    num_receivers = args.num_receivers

    pretraining_lr = args.pretraining_lr
    receiver_lr = args.finetuning_lr
    sender_lr = args.finetuning_lr
    commentary_lr = args.commentary_lr

    batch_size = args.batch_size

    pretraining_epochs = args.pretraining_epochs
    finetuning_epochs = args.finetuning_epochs

    inner_loop_steps = args.inner_loop_steps
    commentary_nn = commentary_networks.idx_commentary_network(num_senders, num_receivers, 16, 16)

    senders = [agents.lstm_sender_agent(feature_size=2049, text_embedding_size=128, vocab_size=2000, lstm_size=128,
                                        lstm_depth=2, feature_embedding_hidden_size=64) for _ in range(num_senders)]
    pretrain_sender = lambda sender: training.pretrain_sender_lstm(sender=sender, path=path + 'sender_pretraining',
                                                                   writer_tag=writer_tag, batch_size=batch_size,
                                                                   num_distractors=num_distractors,
                                                                   num_episodes=pretraining_epochs, lr=pretraining_lr,
                                                                   device=device)
    print('Pretraining senders:')
    senders = [pretrain_sender(sender) for sender in tqdm(senders)]

    receivers = [agents.lstm_receiver_agent(feature_size=2048, text_embedding_size=128, vocab_size=2000, lstm_size=128,
                                            lstm_depth=2, feature_embedding_hidden_size=64, readout_hidden_size=32) for
                 _ in range(num_receivers)]

    pretrain_receiver = lambda receiver: training.pretrain_receiver_lstm(receiver=receiver,
                                                                         num_episodes=pretraining_epochs,
                                                                         path=path + 'receiver_pretraining',
                                                                         writer_tag=writer_tag,
                                                                         batch_size=batch_size,
                                                                         num_distractors=num_distractors,
                                                                         lr=pretraining_lr, device=device)
    print('Pretraining receivers:')
    receivers = [pretrain_receiver(receiver) for receiver in tqdm(receivers)]
    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'pretraining' + '_' + what_agent + '/'
            file_name = what_agent + '_' + str(num) + '.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)

    marl_training.idx_commentary_training_interactive_only(senders=senders, receivers=receivers,
                                                                        commentary_network=commentary_nn,
                                                                        receiver_lr=receiver_lr, sender_lr=sender_lr,
                                                                        commentary_lr=commentary_lr,
                                                                        num_distractors=num_distractors, path=path,
                                                                        writer_tag=writer_tag,
                                                                        num_inner_loop_steps=inner_loop_steps,
                                                                        num_episodes=finetuning_epochs,
                                                                        batch_size=batch_size, device=device)

    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'finetuning' + '_' + what_agent + '/'
            file_name = what_agent + '_' + str(num) + '.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)

    # save commentary_network
    c_network_path = save_path + 'commentary_network/commment.pt'
    os.makedirs(save_path + 'commentary_network')
    torch.save(commentary_nn.state_dict(), c_network_path)

def commentary_weighting_training_lstm128(args):

    key_args = ['num_senders', 'num_receivers', 'num_distractors']
    path = f'results/{args.experiment}/{summarize_key_args(args, key_args)}/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}/'
    path = os.getcwd() + '/' path

    save_path = path + 'saves/'
    for which_training in ['pretraining', 'finetuning']:
        for agent in ['sender', 'receiver']:
            sub_dir = which_training + '_' + agent
            os.makedirs(save_path + sub_dir)

    writer_tag = args.tag
    num_distractors = args.num_distractors

    device = torch.device('cuda:0')
    num_senders = args.num_senders
    num_receivers = args.num_receivers

    pretraining_lr = args.pretraining_lr
    receiver_lr = args.finetuning_lr
    sender_lr = args.finetuning_lr
    commentary_lr = args.commentary_lr

    batch_size = args.batch_size

    pretraining_epochs = args.pretraining_epochs
    finetuning_epochs = args.finetuning_epochs

    inner_loop_steps = args.inner_loop_steps
    commentary_nn = commentary_networks.objects_commentary_network_normalized(num_senders, num_receivers, 64, 2049, 2,
                                                                              2, 64, 4)

    senders = [agents.lstm_sender_agent(feature_size=2049, text_embedding_size=128, vocab_size=2000, lstm_size=128,
                                        lstm_depth=2, feature_embedding_hidden_size=64) for _ in range(num_senders)]
    pretrain_sender = lambda sender: training.pretrain_sender_lstm(sender=sender, path=path + 'sender_pretraining',
                                                                   writer_tag=writer_tag, batch_size=batch_size,
                                                                   num_distractors=num_distractors,
                                                                   num_episodes=pretraining_epochs, lr=pretraining_lr,
                                                                   device=device)
    print('Pretraining senders:')
    senders = [pretrain_sender(sender) for sender in tqdm(senders)]

    receivers = [agents.lstm_receiver_agent(feature_size=2048, text_embedding_size=128, vocab_size=2000, lstm_size=128,
                                            lstm_depth=2, feature_embedding_hidden_size=64, readout_hidden_size=32) for
                 _ in range(num_receivers)]

    pretrain_receiver = lambda receiver: training.pretrain_receiver_lstm(receiver=receiver,
                                                                         num_episodes=pretraining_epochs,
                                                                         path=path + 'receiver_pretraining',
                                                                         writer_tag=writer_tag,
                                                                         batch_size=batch_size,
                                                                         num_distractors=num_distractors,
                                                                         lr=pretraining_lr, device=device)
    print('Pretraining receivers:')
    receivers = [pretrain_receiver(receiver) for receiver in tqdm(receivers)]
    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'pretraining' + '_' + what_agent + '/'
            file_name = what_agent + '_' + str(num) + '.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)

    marl_training.weighted_softmax_commentary_training_interactive_only(senders=senders, receivers=receivers, commentary_network=commentary_nn, receiver_lr=receiver_lr, sender_lr=sender_lr,
                                                          commentary_lr=commentary_lr, num_distractors=num_distractors, path=path, writer_tag=writer_tag, num_inner_loop_steps=inner_loop_steps, num_episodes=finetuning_epochs,
                                                          batch_size=batch_size, device=device)

    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'finetuning' + '_' + what_agent + '/'
            file_name = what_agent + '_' + str(num) + '.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)

    #save commentary_network
    c_network_path = save_path + 'commentary_network/commment.pt'
    os.makedirs(save_path + 'commentary_network')
    torch.save(commentary_nn.state_dict(), c_network_path)


def baseline_population_training_lstm128(args):
    key_args = ['num_senders', 'num_receivers', 'num_distractors']
    path = f'results/{args.experiment}/{summarize_key_args(args, key_args)}/{datetime.now().strftime("%m_%d_%Y,%H:%M:%S")}/'
    path = os.getcwd() + '/' path


    save_path = path + 'saves/'
    for which_training in ['pretraining', 'finetuning']:
        for agent in ['sender', 'receiver']:
            sub_dir = which_training+'_'+agent
            os.makedirs(save_path+sub_dir)

    writer_tag = args.tag
    num_distractors = args.num_distractors

    device = torch.device('cuda:0')
    num_senders = args.num_senders
    num_receivers = args.num_receivers

    pretraining_lr = args.pretraining_lr
    receiver_lr = args.finetuning_lr
    sender_lr = args.finetuning_lr

    batch_size = args.batch_size

    pretraining_epochs = args.pretraining_epochs
    finetuning_epochs = args.finetuning_epochs




    senders = [agents.lstm_sender_agent(feature_size=2049, text_embedding_size=128, vocab_size=2000, lstm_size=128, lstm_depth=2, feature_embedding_hidden_size=64) for _ in range(num_senders)]
    pretrain_sender = lambda sender: training.pretrain_sender_lstm(sender=sender, path=path + 'sender_pretraining',
                                  writer_tag=writer_tag, batch_size=batch_size, num_distractors=num_distractors,
                                  num_episodes=pretraining_epochs, lr=pretraining_lr, device=device)
    print('Pretraining senders:')
    senders = [pretrain_sender(sender) for sender in tqdm(senders)]

    receivers = [agents.lstm_receiver_agent(feature_size=2048, text_embedding_size=128, vocab_size=2000, lstm_size=128, lstm_depth=2, feature_embedding_hidden_size=64, readout_hidden_size=32) for _ in range(num_receivers)]

    pretrain_receiver = lambda receiver: training.pretrain_receiver_lstm(receiver=receiver, num_episodes=pretraining_epochs,
                                      path=path + 'receiver_pretraining', writer_tag=writer_tag,
                                      batch_size=batch_size, num_distractors=num_distractors, lr=pretraining_lr, device=device)
    print('Pretraining receivers:')
    receivers = [pretrain_receiver(receiver) for receiver in tqdm(receivers)]
    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'pretraining'+'_'+what_agent+'/'
            file_name = what_agent + '_' + str(num)+'.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)

    print('Interactive finetuning')
    marl_training.baseline_multiagent_training_interactive_only(senders, receivers, receiver_lr, sender_lr, num_distractors, path+'finetuning',
                                                  writer_tag=writer_tag, num_episodes=finetuning_epochs, batch_size=batch_size, device=device,
                                                  baseline_polyak=0.99)
    for what_agent, networks in zip(['sender', 'receiver'], [senders, receivers]):
        for num, network in enumerate(networks):
            sub_dir = 'finetuning'+'_'+what_agent+'/'
            file_name = what_agent + '_' + str(num)+'.pt'
            network_path = save_path + sub_dir + file_name
            torch.save(network.state_dict(), network_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='scripts running MARL LE experiments')
    parser.add_argument('--experiment')
    parser.add_argument('--pretraining_lr', type=float, default=0.0001)
    parser.add_argument('--finetuning_lr', type=float, default=0.00001)
    parser.add_argument('--num_senders', type=int, default=2)
    parser.add_argument('--num_receivers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--pretraining_epochs', type=int, default=25)
    parser.add_argument('--finetuning_epochs', type=int, default=400)
    parser.add_argument('--tag', default='no_tag_given')
    parser.add_argument('--num_distractors', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fifo_size', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--commentary_lr', type=float, default=0.00001)
    parser.add_argument('--inner_loop_steps', type=int, default=2)

    script_dict = {'baseline_population_training_lstm128':baseline_population_training_lstm128, 'tscl_population_training_lstm128':tscl_population_training_lstm128, 'commentary_weighting_training_lstm128':commentary_weighting_training_lstm128, 'commentary_idx_training_lstm128':commentary_idx_training_lstm128}

    args = parser.parse_args()
    print(vars(args))
    script_dict[args.experiment](args)

