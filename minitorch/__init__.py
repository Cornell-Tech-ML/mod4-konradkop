"""Package contains various functionalities for mathematical operations,
tensor computations, optimization, automatic differentiation, and neural
network layers in the context of deep learning.

Modules included in this package:
---------------------------------
- **testing**: Provides utilities for unit testing and validation.
- **datasets**: Includes datasets and data handling utilities.
- **optim**: Implements optimization algorithms like SGD, Adam, etc.
- **tensor**: Core tensor manipulation functions.
- **nn**: Neural network layers and building blocks.
- **fast_conv**: Optimized convolution operations.
- **tensor_data**: Utilities for tensor data handling.
- **tensor_functions**: High-level tensor functions and operations.
- **tensor_ops**: Tensor-level operations for low-level computation.
- **scalar**: Operations on scalar values.
- **scalar_functions**: Functions for scalar operations.
- **module**: Core module functionality for network components.
- **autodiff**: Automatic differentiation utilities.

This package enables efficient computations for deep learning models, including
tensor manipulations, optimization strategies, and model training utilities.
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .nn import *  # noqa: F401,F403
from .fast_conv import *  # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .fast_ops import *  # noqa: F401,F403
from .cuda_ops import *  # noqa: F401,F403
from . import fast_ops, cuda_ops  # noqa: F401,F403
