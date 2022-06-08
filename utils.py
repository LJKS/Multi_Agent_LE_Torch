import numpy as np
import scipy.stats as stats
import sys

def cycle_dataloader(dataloader):
    while True:
        for x in dataloader:
            yield x

def summarize_test_sentences(org_sentences, test_sentences):
    should_be_sentenecs = ['| but should be | '.join([o_s, t_s]) for o_s, t_s in zip(org_sentences, test_sentences)]
    lines = '  \n'.join(should_be_sentenecs)
    lines.replace('endofsequence', '<eos>').replace('startofsequence', '<start>')
    return lines

def batch2sentences(vocab, idxs):
    idx2word=vocab['idx2word']
    def idxs2sentence(seq):
        seq = seq.numpy()
        sentence = [idx2word[idx] for idx in seq]
        return ' '.join(sentence)

    sentences = [idxs2sentence(seq) for seq in idxs]
    return sentences

class fifo_buffer():
    def __init__(self, size=10):
        self.values = None
        self.timesteps = None
        self.size=10

    def update(self, value, timestep):
        if np.any(self.values == None):
            self.values = np.asarray([value, value])
            self.timesteps = np.asarray([timestep-1, timestep])

        else:
            #update fifo buffers but only keep self.size last values
            self.values = np.append(self.values, value)[-self.size:]
            self.timesteps = np.append(self.timesteps, timestep)[-self.size:]

    def get_t_and_v(self):
        return self.timesteps, self.values

class tscl_helper():
    def __init__(self, num_senders, num_receivers, fifo_size):
        self.num_senders = num_senders
        self.num_receivers = num_receivers
        self.fifo_buffers = [fifo_buffer(size=fifo_size) for _ in range(num_senders*num_receivers)]
        self.task_rewards = np.zeros(shape=(num_senders*num_receivers,), dtype=np.float32)
        self.step=0
        self.create_st2f_idxs()


    def flat_idx(self, sender_idx, receiver_idx):
        flat_idx = sender_idx*self.num_receivers + receiver_idx
        return flat_idx

    def update(self, sender_idx, receiver_idx, value):
        flat_idx = self.flat_idx(sender_idx, receiver_idx)
        self.fifo_buffers[flat_idx].update(value, self.step)
        timesteps, values = self.fifo_buffers[flat_idx].get_t_and_v()
        #zerocenter timesteps for better numeric properties probably
        #use np.polyfit instead, which is much more efficient for small like 10 we have here
        linreg = stats.linregress(x=timesteps-self.step, y=values)
        slope = linreg.slope
        self.task_rewards[flat_idx] = np.abs(slope)
        self.step = self.step + 1

    def sender_target_idx(self, flat_idx):
        return self.st2flat[flat_idx]

    def create_st2f_idxs(self):
        sender_target_idxs = {}
        for si in range(self.num_senders):
            for ri in range(self.num_receivers):
                sender_target_idxs[self.flat_idx(si, ri)] = (si, ri)
        self.st2flat = sender_target_idxs

    def sample_epsilon_greedy(self, epsilon):
        rand = np.random.uniform(0,1,size=(1,))[0]
        sender = None
        receiver = None
        f_idx = None
        if rand < epsilon:
            #random
            f_idx = np.random.randint(0,self.num_receivers*self.num_senders)
        else:
            f_idx = np.argmax(self.task_rewards)
        sender, receiver = self.sender_target_idx(f_idx)
        return sender, receiver








