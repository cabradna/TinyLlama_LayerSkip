import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        process_group=None,
    ):
        super().__init__()
        if reduction not in ["mean", "none"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.process_group = process_group

    def forward(self, input, target):
        # Ensure inputs are on GPU
        if not input.is_cuda:
            input = input.cuda()
        if not target.is_cuda:
            target = target.cuda()

        if len(input.shape) == 3:
            batch_size, seq_len, vocab_size = input.shape
            input = input.view(-1, vocab_size)
            target = target.view(-1)

        if self.process_group is None:
            # Standard cross entropy with label smoothing
            with torch.cuda.amp.autocast(enabled=True):  # Enable automatic mixed precision
                loss = F.cross_entropy(
                    input,
                    target,
                    ignore_index=self.ignore_index,
                    reduction='none',
                    label_smoothing=self.label_smoothing
                )
        else:
            # Distributed version with tensor parallel
            world_size = torch.distributed.get_world_size(self.process_group)
            rank = torch.distributed.get_rank(self.process_group)
            vocab_size = input.size(-1)
            
            # Handle distributed vocabulary
            vocab_start_index = rank * vocab_size
            vocab_end_index = (rank + 1) * vocab_size
            
            # Adjust labels for local vocabulary
            ignored_mask = target == self.ignore_index
            labels_local = torch.where(ignored_mask, target, target - vocab_start_index)
            
            # Calculate local loss using CUDA operations
            with torch.cuda.amp.autocast(enabled=True):
                log_probs = F.log_softmax(input, dim=-1)
                
                if self.label_smoothing > 0:
                    smooth_loss = -log_probs.mean(dim=-1) * self.label_smoothing
                    main_loss = F.nll_loss(
                        log_probs, labels_local, 
                        ignore_index=self.ignore_index,
                        reduction='none'
                    ) * (1 - self.label_smoothing)
                    loss = main_loss + smooth_loss
                else:
                    loss = F.nll_loss(
                        log_probs, labels_local,
                        ignore_index=self.ignore_index,
                        reduction='none'
                    )
            
            # Synchronize across processes
            torch.distributed.all_reduce(
                loss,
                op=torch.distributed.ReduceOp.SUM,
                group=self.process_group
            )

        if self.reduction == "mean":
            valid_elements = (target != self.ignore_index).sum()
            return loss.sum() / valid_elements
        return loss
