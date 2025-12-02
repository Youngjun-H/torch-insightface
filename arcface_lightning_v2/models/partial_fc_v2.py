"""
Partial FC V2 for distributed training
Lightning 환경에 맞게 수정
"""

from typing import Callable

import torch
from torch import distributed
from torch.nn.functional import linear, normalize


class PartialFC_V2(torch.nn.Module):
    """
    https://arxiv.org/abs/2203.15565
    A distributed sparsely updating variant of the FC layer, named Partial FC (PFC).
    """

    _version = 2

    def __init__(
        self,
        margin_loss: Callable,
        embedding_size: int,
        num_classes: int,
        sample_rate: float = 1.0,
        fp16: bool = False,
    ):
        """
        Parameters:
        -----------
        embedding_size: int
            The dimension of embedding, required
        num_classes: int
            Total number of classes, required
        sample_rate: float
            The rate of negative centers participating in the calculation, default is 1.0.
        """
        super(PartialFC_V2, self).__init__()

        # Lightning에서는 distributed가 나중에 초기화될 수 있으므로
        # lazy initialization을 사용
        self._initialized = False
        self.margin_loss = margin_loss
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.fp16 = fp16

        # 초기화는 setup_distributed에서 수행
        self.rank = None
        self.world_size = None
        self.num_local = None
        self.class_start = None
        self.num_sample = None
        self.weight = None
        self.last_batch_size = 0
        self.dist_cross_entropy = None

    def setup_distributed(self):
        """Distributed 환경이 준비된 후 호출"""
        if self._initialized:
            return

        if not distributed.is_initialized():
            # Single GPU 또는 CPU인 경우
            self.rank = 0
            self.world_size = 1
        else:
            self.rank = distributed.get_rank()
            self.world_size = distributed.get_world_size()

        self.dist_cross_entropy = DistCrossEntropy()
        self.num_local = self.num_classes // self.world_size + int(
            self.rank < self.num_classes % self.world_size
        )
        self.class_start = self.num_classes // self.world_size * self.rank + min(
            self.rank, self.num_classes % self.world_size
        )
        self.num_sample = int(self.sample_rate * self.num_local)

        # margin_loss
        if isinstance(self.margin_loss, Callable):
            self.margin_softmax = self.margin_loss
        else:
            raise ValueError("margin_loss must be callable")

        # Weight 초기화
        self.weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (self.num_local, self.embedding_size))
        )

        self._initialized = True

    def sample(self, labels, index_positive, device):
        """Sample negative classes"""
        with torch.no_grad():
            positive = torch.unique(labels[index_positive], sorted=True).to(device)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local], device=device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1].to(device)
                index = index.sort()[0].to(device)
            else:
                index = positive
            self.weight_index = index

            labels[index_positive] = torch.searchsorted(index, labels[index_positive])

        weight = self.weight
        if weight.device != device:
            weight = weight.to(device=device)
        return weight[self.weight_index]

    def forward(
        self,
        local_embeddings: torch.Tensor,
        local_labels: torch.Tensor,
    ):
        """
        Parameters:
        ----------
        local_embeddings: torch.Tensor
            feature embeddings on each GPU(Rank).
        local_labels: torch.Tensor
            labels on each GPU(Rank).
        Returns:
        -------
        loss: torch.Tensor
            pass
        """
        # Lazy initialization
        if not self._initialized:
            self.setup_distributed()

        local_labels.squeeze_()
        local_labels = local_labels.long()

        batch_size = local_embeddings.size(0)
        if self.last_batch_size == 0:
            self.last_batch_size = batch_size
        assert self.last_batch_size == batch_size, (
            f"last batch size do not equal current batch size: "
            f"{self.last_batch_size} vs {batch_size}"
        )

        device = local_embeddings.device
        dtype = local_embeddings.dtype  # AMP 사용 시 dtype 일치를 위해 필요

        if self.world_size > 1 and distributed.is_initialized():
            # Distributed training
            # local_embeddings의 dtype을 사용하여 gather_list 생성 (AMP 호환성)
            _gather_embeddings = [
                torch.zeros(
                    (batch_size, self.embedding_size), dtype=dtype, device=device
                )
                for _ in range(self.world_size)
            ]
            _gather_labels = [
                torch.zeros(batch_size, dtype=torch.long, device=device)
                for _ in range(self.world_size)
            ]
            _list_embeddings = AllGather(local_embeddings, *_gather_embeddings)
            distributed.all_gather(_gather_labels, local_labels)

            embeddings = torch.cat(_list_embeddings)
            labels = torch.cat(_gather_labels)
        else:
            # Single GPU or CPU
            embeddings = local_embeddings
            labels = local_labels

        labels = labels.view(-1, 1)
        index_positive = (self.class_start <= labels) & (
            labels < self.class_start + self.num_local
        )
        labels[~index_positive] = -1
        labels[index_positive] -= self.class_start

        if self.sample_rate < 1:
            weight = self.sample(labels, index_positive, device)
        else:
            weight = self.weight

        # ⚠️ 중요: PartialFC weight는 _modules에 등록되지 않아 Lightning이 자동으로 GPU로 이동시키지 않음
        # 따라서 명시적으로 device와 dtype을 맞춰야 함
        if weight.device != device:
            weight = weight.to(device=device)
        if weight.dtype != embeddings.dtype:
            weight = weight.to(dtype=embeddings.dtype)

        norm_embeddings = normalize(embeddings)
        norm_weight_activated = normalize(weight)
        logits = linear(norm_embeddings, norm_weight_activated)
        logits = logits.clamp(-1, 1)

        logits = self.margin_softmax(logits, labels)
        loss = self.dist_cross_entropy(logits, labels)
        return loss


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)

        # local to global (if distributed)
        if distributed.is_initialized():
            distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        if distributed.is_initialized():
            distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)

        index = torch.where(label != -1)[0]
        # loss - logits와 같은 dtype 사용 (AMP 호환성)
        loss = torch.zeros(batch_size, 1, dtype=logits.dtype, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        if distributed.is_initialized():
            distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        # one_hot도 logits와 같은 dtype 사용 (AMP 호환성)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)],
            dtype=logits.dtype,
            device=logits.device,
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)


class AllGatherFunc(torch.autograd.Function):
    """AllGather op with gradient backward"""

    @staticmethod
    def forward(ctx, tensor, *gather_list):
        gather_list = list(gather_list)
        if distributed.is_initialized():
            # dtype 일치 보장 (AMP 사용 시 중요)
            # tensor의 dtype과 device를 기준으로 모든 gather 텐서를 변환
            tensor_dtype = tensor.dtype
            tensor_device = tensor.device

            # 모든 gather 텐서가 tensor와 같은 dtype과 device를 가지도록 보장
            # to() 메서드는 새로운 텐서를 반환하므로 리스트에 다시 할당해야 함
            for i in range(len(gather_list)):
                if (
                    gather_list[i].dtype != tensor_dtype
                    or gather_list[i].device != tensor_device
                ):
                    gather_list[i] = gather_list[i].to(
                        dtype=tensor_dtype, device=tensor_device
                    )

            # distributed.all_gather 호출
            distributed.all_gather(gather_list, tensor)
        else:
            # Single GPU: just copy
            gather_list[0] = tensor
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        grad_list = list(grads)
        if distributed.is_initialized():
            rank = distributed.get_rank()
            grad_out = grad_list[rank]

            dist_ops = [
                (
                    distributed.reduce(
                        grad_out, rank, distributed.ReduceOp.SUM, async_op=True
                    )
                    if i == rank
                    else distributed.reduce(
                        grad_list[i], i, distributed.ReduceOp.SUM, async_op=True
                    )
                )
                for i in range(distributed.get_world_size())
            ]
            for _op in dist_ops:
                _op.wait()

            grad_out *= len(grad_list)  # cooperate with distributed loss function
        else:
            grad_out = grad_list[0]

        return (grad_out, *[None for _ in range(len(grad_list))])


AllGather = AllGatherFunc.apply
