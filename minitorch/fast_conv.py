from typing import Tuple, TypeVar, Any

from numba import prange
from numba import njit as _njit
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Just-In-Time (JIT) compiler decorator for optimizing the given function.

    This function wraps the provided function `fn` with a JIT compiler, which can
    optimize its performance by compiling it to machine code. The JIT compilation
    can provide significant speedup for numerical and array-based computations.

    The decorator uses `inline="always"` to attempt to inline the function during
    the JIT compilation, optimizing its execution further. Additional keyword arguments
    can be passed to configure other JIT compilation options.

    Args:
    ----
    fn (Fn): The function to be JIT-compiled.
    **kwargs (Any): Additional keyword arguments passed to the JIT compiler, e.g., optimization options.

    Returns:
    -------
    Fn: The JIT-compiled version of the input function.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # # # TODO: Implement for Task 4.1.
    # # raise NotImplementedError("Need to implement for Task 4.1")
    for out_idx in prange(out_size):
        # Directly compute batch, output channel, and width indices
        current_batch = out_idx // (out_channels * out_width)
        remaining = out_idx % (out_channels * out_width)
        current_out_channel = remaining // out_width
        current_width = remaining % out_width

        val = 0.0

        for current_in_channel in range(in_channels):
            input_base = current_batch * s1[0] + current_in_channel * s1[1]
            weight_base = current_out_channel * s2[0] + current_in_channel * s2[1]

            for k in range(kw):
                weight_idx = weight_base + k * s2[2]
                w = weight[weight_idx]

                input_offset = (
                    current_width - (kw - k - 1) if reverse else current_width + k
                )
                if 0 <= input_offset < width:
                    input_idx = input_base + input_offset * s1[2]
                    inc = input[input_idx]
                    val += w * inc

        out[out_idx] = val


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradients of the input and weight for the 1D convolution operation.

        This method computes the gradients of the input and weight tensors with respect
        to the loss using the chain rule, based on the stored values from the forward pass.

        The backward pass involves computing:
        - The gradient with respect to the input tensor (`grad_input`).
        - The gradient with respect to the weight tensor (`grad_weight`).

        Args:
        ----
            ctx (Context): The context object storing information from the forward pass,
                            including the input and weight tensors.
            grad_output (Tensor): The gradient of the loss with respect to the output of
                                   the convolution operation.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple containing the gradient with respect to the
                                    input tensor (`grad_input`) and the gradient with
                                    respect to the weight tensor (`grad_weight`).

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )

        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # # TODO: Implement for Task 4.2.
    # raise NotImplementedError("Need to implement for Task 4.2")
    batch, out_channels, out_height, out_width = out_shape
    _, in_channels, height, width = input_shape
    _, _, kh, kw = weight_shape

    s1_b, s1_c, s1_h, s1_w = input_strides
    s2_oc, s2_ic, s2_kh, s2_kw = weight_strides
    # s_out_b, s_out_c, s_out_h, s_out_w = out_strides
    for out_idx in prange(out_size):
        # Compute indices for batch, channel, height, and width
        b = out_idx // (out_channels * out_height * out_width)
        remaining = out_idx % (out_channels * out_height * out_width)
        oc = remaining // (out_height * out_width)
        remaining = remaining % (out_height * out_width)
        h = remaining // out_width
        w = remaining % out_width

        result = 0.0

        # Iterate over input channels
        for ic in range(in_channels):
            # Base indices for input and weight
            input_base = b * s1_b + ic * s1_c
            weight_base = oc * s2_oc + ic * s2_ic

            for k_h in range(kh):
                for k_w in range(kw):
                    # Reverse indexing logic
                    input_h = h + k_h if not reverse else h - k_h
                    input_w = w + k_w if not reverse else w - k_w

                    # Check bounds for input indices
                    if 0 <= input_h < height and 0 <= input_w < width:
                        # Compute input and weight positions
                        input_pos = input_base + input_h * s1_h + input_w * s1_w
                        weight_pos = weight_base + k_h * s2_kh + k_w * s2_kw

                        # Accumulate the result
                        result += input[input_pos] * weight[weight_pos]

        # Write the result to the output tensor
        out[out_idx] = result


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for the 2D convolution operation. Computes the gradients
        with respect to the input and the weights.

        This method implements the backpropagation for the convolution operation.
        It computes the gradients of the input and weight tensors by performing
        the necessary convolutions with the gradient of the output.

        The backward computation for 2D convolution involves:
        1. Computing the gradient with respect to the weight tensor by convolving
           the gradient of the output with the input.
        2. Computing the gradient with respect to the input tensor by convolving
           the gradient of the output with the transposed weights.

        The gradients are calculated by applying the chain rule of backpropagation
        and using the convolution operation in reverse.

        Args:
        ----
            ctx : Context
                The context object that contains saved values from the forward pass
                (input and weight) which are needed for the backward pass.

            grad_output : Tensor
                The gradient of the loss with respect to the output tensor. This is
                used to compute the gradients with respect to the input and weight
                tensors.

        Returns:
        -------
            Tuple[Tensor, Tensor] :
                - grad_input (Tensor): The gradient of the loss with respect to the input tensor.
                - grad_weight (Tensor): The gradient of the loss with respect to the weight tensor.

        """
        # Retrieve the saved input and weight tensors from the forward pass
        input, weight = ctx.saved_values

        # Get the shapes of the input and weight tensors
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        # Initialize the gradient for the weight tensor with zeros
        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))

        # Permute the input and grad_output to align dimensions for convolution
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)

        # Compute the gradient of the weight by convolving the input with the grad_output
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )

        # Permute grad_weight back to the original shape
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        # Initialize the gradient for the input tensor with zeros
        grad_input = input.zeros((batch, in_channels, h, w))

        # Permute the weight to prepare for convolution with grad_output
        new_weight = weight.permute(1, 0, 2, 3)

        # Compute the gradient of the input by convolving the grad_output with the transposed weight
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )

        # Return both gradients: grad_input and grad_weight
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
