# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Communications utilities."""


import torch
import torch.distributed as dist

from megatron.core import mpu



# TODO: use functions from megatron/p2p
def recv_from_prev_pipeline_rank_(recv_buffer=None):
    """Receive from previous pipeline stage and update the
    input buffer inplace."""
    if not mpu.is_pipeline_first_stage():
        assert recv_buffer is not None
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv, recv_buffer,
            mpu.get_pipeline_model_parallel_prev_rank())
        reqs = torch.distributed.batch_isend_irecv([recv_prev_op])
        for req in reqs:
            req.wait()
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()



# TODO: use functions from megatron/p2p
def send_to_next_pipeline_rank(tensor=None):
    """Send output to the next pipeline stage."""
    if not mpu.is_pipeline_last_stage():
        assert tensor is not None
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor,
            mpu.get_pipeline_model_parallel_next_rank())
        reqs = torch.distributed.batch_isend_irecv([send_next_op])
        for req in reqs:
            req.wait()
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()


def recv_list_from_prev_pipeline_rank(recv_buffers):
    if not mpu.is_pipeline_first_stage():
        assert recv_buffers is not None and type(recv_buffers) is list
        recv_prev_ops = [torch.distributed.P2POp(
            torch.distributed.irecv, recv_buffer,
            mpu.get_pipeline_model_parallel_prev_rank()) for recv_buffer in recv_buffers]
        reqs = torch.distributed.batch_isend_irecv(recv_prev_ops)
        for req in reqs:
            req.wait()
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()


def send_list_to_next_pipeline_rank(tensors):
    if not mpu.is_pipeline_last_stage():
        assert tensors is not None and type(tensors) is list
        send_next_ops = [torch.distributed.P2POp(
            torch.distributed.isend, tensor,
            mpu.get_pipeline_model_parallel_next_rank()) for tensor in tensors]
        reqs = torch.distributed.batch_isend_irecv(send_next_ops)
        for req in reqs:
            req.wait()
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()


def _is_cuda(tensor):
    """Check if a tensor is not none and is cuda."""
    assert tensor is not None
    assert tensor.is_cuda



def _is_cuda_contiguous(tensor):
    """Check if a tensor is not none, is cuda, and is contiguous."""
    _is_cuda(tensor)
    assert tensor.is_contiguous()



def broadcast_from_last_pipeline_stage(size, dtype, tensor=None):
    """Broadcast a tensor from last pipeline stage to all ranks."""

    is_last_stage = mpu.is_pipeline_last_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if mpu.is_pipeline_first_stage() and is_last_stage:
        return tensor

    if is_last_stage:
        _is_cuda_contiguous(tensor)
    else:
        tensor = torch.empty(size,
                             dtype=dtype,
                             device=torch.cuda.current_device())
    # Get the group and corresponding source rank.
    src = mpu.get_pipeline_model_parallel_last_rank()
    group = mpu.get_pipeline_model_parallel_group()
    torch.distributed.broadcast(tensor, src, group)

    return tensor


def broadcast_from_first_pipeline_stage(size, dtype, tensor=None):
    """Broadcast a tensor from last pipeline stage to all ranks."""

    is_first_stage = mpu.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if mpu.is_pipeline_last_stage() and is_first_stage:
        return tensor

    if is_first_stage:
        _is_cuda_contiguous(tensor)
    else:
        tensor = torch.empty(size,
                             dtype=dtype,
                             device=torch.cuda.current_device())
    # Get the group and corresponding source rank.
    src = mpu.get_pipeline_model_parallel_first_rank()
    group = mpu.get_pipeline_model_parallel_group()
    torch.distributed.broadcast(tensor, src, group)

    return tensor


def broadcast_from_last_to_first_pipeline_stage(size, dtype, tensor=None):
    """Broadcast tensor values from last stage into the first stage."""

    is_last_stage = mpu.is_pipeline_last_stage()
    is_first_stage = mpu.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if is_first_stage and is_last_stage:
        return tensor
    # Only first and last stage pipeline stages need to be involved.
    if is_last_stage or is_first_stage:
        if is_last_stage:
            _is_cuda_contiguous(tensor)
        else:
            tensor = torch.empty(size,
                                 dtype=dtype,
                                 device=torch.cuda.current_device())
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_pipeline_endpoint_group()
        # Broadcast from last stage into the first stage.
        torch.distributed.broadcast(tensor, src, group)
    else:
        tensor = None

    return tensor



def copy_from_last_to_first_pipeline_stage(size, dtype, tensor=None):
    """Copy tensor values from last stage into the first stage.
    Note that the input tensor is updated in place."""

    is_last_stage = mpu.is_pipeline_last_stage()
    is_first_stage = mpu.is_pipeline_first_stage()
    # If first stage and last state are the same, then there is no
    # pipeline parallelism and no need to communicate.
    if is_first_stage and is_last_stage:
        return
    # Only first and last stage pipeline stages need to be involved.
    if is_last_stage or is_first_stage:
        _is_cuda(tensor)
        is_contiguous = tensor.is_contiguous()
        src = mpu.get_pipeline_model_parallel_last_rank()
        group = mpu.get_pipeline_endpoint_group()
        if is_contiguous:
            tensor_ = tensor
        else:
            if is_last_stage:
                tensor_ = tensor.contiguous()
            else:
                tensor_ = torch.empty(size,
                                      dtype=dtype,
                                      device=torch.cuda.current_device())
        # Broadcast from last stage into the first stage.
        torch.distributed.broadcast(tensor_, src, group)
        # Update the first stage tensor
        if is_first_stage and not is_contiguous:
            tensor[...] = tensor_


def get_exit_stages():
    early_exit_stage_ids = mpu.get_early_exit_stages()
    last_stage_id = mpu.get_pipeline_model_parallel_world_size() - 1
    if last_stage_id not in early_exit_stage_ids:
        return list(early_exit_stage_ids + [last_stage_id])
    return early_exit_stage_ids


EXIT=1
CONTINUE=0

def send_token_and_probs_to_first_pipeline_stage(inference_params, token_tensor=None, prob_tensor=None, is_final=False):
    signal_tensor = torch.empty(1, dtype=torch.int8, device=torch.cuda.current_device())
    if inference_params.has_early_exited or is_final:
        signal_tensor[0] = EXIT
        _is_cuda(token_tensor)
        _is_cuda(prob_tensor)
    else:
        signal_tensor[0] = CONTINUE
    dist.send(tensor=signal_tensor, dst=0, group=mpu.get_pipeline_model_parallel_group())
    if inference_params.has_early_exited or is_final:
        dist.send(tensor=token_tensor, dst=0, group=mpu.get_pipeline_model_parallel_group())
        dist.send(tensor=prob_tensor, dst=0, group=mpu.get_pipeline_model_parallel_group())


def recv_token_and_probs(inference_params, token_tensor_buffer, prob_tensor_buffer):

    is_contiguous = token_tensor_buffer.is_contiguous()
    if is_contiguous:
        token_tensor_ = token_tensor_buffer
        prob_tensor_ = prob_tensor_buffer
    else:
        token_tensor_ = torch.empty(token_tensor_buffer.shape[0],
                                dtype=torch.int64,
                                device=torch.cuda.current_device())
        prob_tensor_ = torch.empty(prob_tensor_buffer.shape[0],
                                dtype=torch.float32,
                                device=torch.cuda.current_device())

    # if first stage has early exit, get tensor directly
    if mpu.has_early_exit():
        if inference_params.has_early_exited:
            assert inference_params.tokens is not None
            token_tensor_buffer[...] = inference_params.tokens
            prob_tensor_buffer[...] = inference_params.probs
            return

    exit_stages = get_exit_stages()
    if exit_stages[0] == 0:
        exit_stages.pop(0)
    signal_tensor = torch.empty(1, dtype=torch.int8, device=torch.cuda.current_device())

    # get tensor from subsequent stages one by one
    for stage_id in exit_stages:
        dist.recv(tensor=signal_tensor, src=stage_id, group=mpu.get_pipeline_model_parallel_group())
        if signal_tensor[0] == EXIT:
            dist.recv(tensor=token_tensor_, src=stage_id, group=mpu.get_pipeline_model_parallel_group())
            dist.recv(tensor=prob_tensor_, src=stage_id, group=mpu.get_pipeline_model_parallel_group())
            break

    if not is_contiguous:
        token_tensor_buffer[...] = token_tensor_
        prob_tensor_buffer[...] = prob_tensor_

def broadcast_tensor(size, dtype, tensor=None, rank=0):
    """ Given size and type of a tensor on all ranks and the tensor value
        only on a specific rank, broadcast from that rank to all other ranks.
    """

    if torch.distributed.get_rank() == rank:
        _is_cuda_contiguous(tensor)
    else:
        tensor = torch.empty(size,
                             dtype=dtype,
                             device=torch.cuda.current_device())

    torch.distributed.broadcast(tensor, rank)

    return tensor



def broadcast_list(size, dtype, list_values=None, rank=0):
    """Broadcast a list of values with a given type."""

    tensor = None
    if torch.distributed.get_rank() == rank:
        tensor = torch.tensor(list_values, dtype=dtype,
                              device=torch.cuda.current_device())

    return broadcast_tensor(size, dtype, tensor=tensor, rank=rank)



def broadcast_int_list(size, int_list=None, rank=0):
    """Broadcast a list of interger values."""

    return broadcast_list(size, torch.int64, list_values=int_list, rank=rank)



def broadcast_float_list(size, float_list=None, rank=0):
    """Broadcast a list of float values."""

    return broadcast_list(size, torch.float32, list_values=float_list,
                          rank=rank)
