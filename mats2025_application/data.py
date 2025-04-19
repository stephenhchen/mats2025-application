from itertools import islice, product
from typing import Generator
import numpy as np
import torch
from transformer_lens import HookedTransformer
from torch.utils.data.dataset import IterableDataset
from mats2025_application.utils import Mess3

class Mess3Dataset(IterableDataset):
    """A thin wrapper around Mess3."""
    def __init__(self, mess3: Mess3, n_ctx: int, max_seq: int, return_belief_dist=False):
        assert hasattr(mess3, "msp"), "Compute MSP before creating dataset"
        self.n_ctx = n_ctx
        self.mess3 = mess3
        self._mess3_setup()
        self.max_seq = max_seq
        self.return_belief_dist = return_belief_dist
        self._seq_counter = 0

    def _mess3_setup(self,):
        self.vocab_len = self.mess3.vocab_len
        self.num_states = self.mess3.num_states
        self.transition_matrix = self.mess3.transition_matrix

    def reset_counter(self):
        self._seq_counter = 0
    
    def __iter__(self):

        n_ctx = self.n_ctx+1 if not self.return_belief_dist else self.n_ctx
        while self._seq_counter < self.max_seq:
            self._seq_counter += 1
            seq = [emission for emission in self.mess3.yield_emissions(n_ctx)]      #  

            if self.return_belief_dist:
                input_seq = torch.tensor(seq)
                output_seq = torch.empty([])
                beliefs = self.mess3.msp.path_to_beliefs(seq)
                beliefs = torch.tensor(beliefs)
            else:
                input_seq = torch.LongTensor(seq[:-1])
                output_seq = torch.LongTensor(seq[1:])
                beliefs = torch.empty([])

            yield input_seq, output_seq, beliefs

    def generate_histories_with_beliefs(self) -> tuple[torch.tensor, np.array]:
        all_seq, all_beliefs = zip(*[(seq, belief) for (seq, _, belief) in iter(self)])
        return torch.vstack(all_seq), np.vstack(all_beliefs)


def compute_activations_for_all_subsequences(model: HookedTransformer,
                                             seq_samples: torch.tensor,
                                             device: str, n_batches=100) -> np.array:
    model.eval()
    n_seq = seq_samples.shape[0]
    model = model.to(device)
    seq_samples = seq_samples.to(device)
    indices = np.linspace(0, n_seq, n_batches).round().astype("int32")

    all_subseq_activ = []
    for i in range(len(indices)-1):
        left_idx, right_idx = indices[i], indices[i+1]
        _, activ_cache = model.run_with_cache(seq_samples[left_idx: right_idx])
        activ = activ_cache["ln_final.hook_normalized"].detach()    # (bs, 10, 64)
        activ_subseq = activ.reshape(-1, model.cfg.d_model)         # (bs*10, 64)
        all_subseq_activ.append(activ_subseq.cpu().numpy())

    return np.vstack(all_subseq_activ)