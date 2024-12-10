from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")
    new_height = height // kh
    new_width = width // kw
    output = (
        input.contiguous()
        .view(batch, channel, new_height, kh, new_width, kw)
        .permute(0, 1, 2, 4, 3, 5)
    )
    output = output.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return (output, new_height, new_width)


# TODO: Implement for Task 4.3.


# - avgpool2d: Tiled average pooling 2D
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
    -------
        :class:`Tensor` : pooled tensor

    """
    batch, channel, height, width = input.shape

    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.mean(dim=4).view(batch, channel, new_height, new_width)

    return pooled


max_reduce = FastOps.reduce(operators.max, -1e9)


# - argmax: Compute the argmax as a 1-hot tensor
def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
    -------
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


# - Max: New Function for max operator
class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int) -> Tensor:
        """Forward of max should perform max reduction.

        Args:
        ----
            ctx (Context): Context to save information for backward.
            input (Tensor): Input tensor.
            dim (int): Dimension to perform the max reduction.

        Returns:
        -------
            Tensor: Result of max reduction along the specified dimension.

        """
        # Perform max reduction along the specified dimension
        max_values, indices = max_reduce(input, dim)  # Get max values and their indices
        ctx.save_for_backward(indices, dim)  # Save indices and dim for backward pass
        return max_values  # type: ignore

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Backward of max should be argmax (see above)"""
        input, dim = ctx.saved_values
        res = argmax(input, dim)
        return grad_output * res


# - max: Apply max reduction
max = Max.apply


# - softmax: Compute the softmax as a tensor


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
    ----
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
    -------
        :class:`Tensor` : softmax tensor

    """
    # Step 1: To prevent overflow, subtract the maximum value of the input tensor
    # along the specified dimension. This ensures stability during exponentiation.
    input_stable = input - max_reduce(input, dim)

    # Step 2: Compute the exponential of each element in the stable input tensor.
    # This transforms the values into a non-linear scale, as required for the softmax.
    e = input_stable.exp()

    # Step 3: Compute the sum of the exponentials along the specified dimension.
    # This sum is used as the denominator in the softmax formula.
    t = e.sum(dim=dim)

    # Step 4: Compute the softmax by dividing each exponentiated element by the
    # sum of exponentials along the specified dimension.
    return e / t


# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
    ----
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
    -------
        :class:`Tensor` : log of softmax tensor

    """
    # Step 1: Find the maximum value along the specified dimension to prevent overflow
    # when computing the exponentials. This step ensures numerical stability.
    m = max_reduce(input, dim)

    # Step 2: Subtract the maximum value from the input tensor to ensure numerical stability.
    # Then exponentiate the result to compute e^(input - max_value).
    e = (input - m).exp()

    # Step 3: Compute the sum of the exponentiated values along the specified dimension.
    # This sum is used in the denominator of the log-softmax formula.
    s = e.sum(dim=dim)

    # Step 4: Compute the log of the sum of exponentials, subtracting the maximum value
    # for stability. Finally, subtract the log-sum-exp value from the input tensor.
    return input - s.log() - m


# - maxpool2d: Tiled max pooling 2D
def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D

    Args:
    ----
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
    -------
        :class:`Tensor` : pooled tensor

    """
    batch, channel, height, width = input.shape

    tiled, new_height, new_width = tile(input, kernel)
    pooled = max_reduce(tiled, 4).view(batch, channel, new_height, new_width)
    return pooled


# - dropout: Dropout positions based on random noise, include an argument to turn off
def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
    -------
        :class:`Tensor` : tensor with random positions dropped out

    """
    if ignore:
        return input
    ratios = rand(input.shape)
    return input * (ratios > rate)
