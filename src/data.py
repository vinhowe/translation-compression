from dataclasses import dataclass

import torch


@dataclass
class UniformDataset:
    vocab_size: int
    seq_len: int
    size: int
    seed: int

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)
        return torch.randint(
            low=0, high=self.vocab_size, size=(self.seq_len,), generator=gen
        )


@dataclass
class UniformBatchDataLoader:
    B: int
    T: int
    vocab_size: int
    seed: int
    process_rank: int = 0
    num_processes: int = 1
    return_compartment_ids: bool = False
    pin_memory: bool = True

    def __post_init__(self) -> None:
        self._pin = bool(self.pin_memory and torch.cuda.is_available())
        self._gen = torch.Generator()
        # ensure different stream per rank; deterministic across runs
        self._gen.manual_seed(int(self.seed) + int(self.process_rank))

    def reset(self) -> None:
        # re-seed to initial state for deterministic epoch resets if needed
        self._gen.manual_seed(int(self.seed) + int(self.process_rank))

    def next_batch(self):
        x = torch.randint(
            0,
            int(self.vocab_size),
            (int(self.B), int(self.T)),
            generator=self._gen,
            dtype=torch.long,
            pin_memory=self._pin,
        )
        y = torch.empty_like(x)
        if self.T > 0:
            y[:, :-1] = x[:, 1:]
            y[:, -1] = -1
        else:
            y.fill_(-1)
        if self.return_compartment_ids:
            # single compartment id 0 by default for all tokens
            cids = torch.zeros(
                (int(self.B), int(self.T)), dtype=torch.long, pin_memory=self._pin
            )
            return x, y, cids
        return x, y
