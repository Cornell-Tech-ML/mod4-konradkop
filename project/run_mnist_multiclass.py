import os.path
# Set HOME environment variable for Windows
os.environ['HOME'] = os.path.expanduser('~')

from mnist import MNIST


import minitorch

mndata = MNIST("./data/")
images, labels = mndata.load_training()

BACKEND = minitorch.TensorBackend(minitorch.FastOps)
BATCH = 16

# Number of classes (10 digits)
C = 10

# Size of images (height and width)
H, W = 28, 28


def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw)
        self.bias = RParam(out_channels, 1, 1)

    def forward(self, input):
        # Direct convolution with the weights and add bias efficiently
        weight_value = self.weights.value
        bias_value = self.bias.value

        # Perform convolution followed by adding the bias
        conv_result = minitorch.conv2d(input, weight_value)

        # Adding the bias to the convolution result (broadcasting is handled automatically)
        return conv_result + bias_value
        return minitorch.conv2d(input, self.weights.value) + self.bias.value
        # TODO: Implement for Task 4.5.
        # raise NotImplementedError("Need to implement for Task 4.5")


class Network(minitorch.Module):
    """
    A Convolutional Neural Network (CNN) for MNIST classification based on the LeNet architecture.

    This model follows the below procedure:
    1. Apply a 2D convolution with 4 output channels and a 3x3 kernel, followed by a ReLU activation. 
       The result is stored in `self.mid`.
    2. Apply another 2D convolution with 8 output channels and a 3x3 kernel, followed by a ReLU activation. 
       The result is stored in `self.out`.
    3. Perform 2D pooling (either average or max) with a 4x4 kernel to downsample the feature map.
    4. Flatten the resulting tensor to a shape of `[batch_size x 392]`.
    5. Pass the flattened tensor through a fully connected (linear) layer to reduce the size to 64, 
       followed by a ReLU activation and dropout with a rate of 25%.
    6. Apply a second linear layer to map the tensor to the number of output classes (`C`).
    7. Apply a log-softmax operation over the class dimension to generate log probabilities for classification.

    Attributes:
    ----------
        mid (:class:`Tensor`): Intermediate result after the first convolution layer.
        out (:class:`Tensor`): Intermediate result after the second convolution layer.
        classes (int): Number of output classes (`C`).
        conv1 (:class:`Conv2d`): First 2D convolutional layer.
        conv2 (:class:`Conv2d`): Second 2D convolutional layer.
        linear1 (:class:`Linear`): First fully connected layer.
        linear2 (:class:`Linear`): Second fully connected layer for classification.
    """

    def __init__(self):
        super().__init__()

        # Attributes for visualization
        self.mid = None
        self.out = None
        self.classes = C  # Set the number of classes dynamically.

        # Define the layers of the network.
        self.conv1 = Conv2d(1, 4, 3, 3)  # Input channels: 1, Output channels: 4, Kernel: 3x3
        self.conv2 = Conv2d(4, 8, 3, 3)  # Input channels: 4, Output channels: 8, Kernel: 3x3
        self.linear1 = Linear(392, 64)  # Fully connected layer: Input size 392, Output size 64
        self.linear2 = Linear(64, self.classes)  # Fully connected layer: Input size 64, Output size C

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
        ----
            x (:class:`Tensor`): Input tensor with shape `[batch_size x 1 x height x width]`, 
                                 where `height` and `width` are the dimensions of the MNIST images.

        Returns:
        -------
            :class:`Tensor`: Output tensor with shape `[batch_size x C]`, where `C` is the number of classes.
        """

        # Apply the first convolutional layer, followed by ReLU activation.
        self.mid = self.conv1.forward(x).relu()

        # Apply the second convolutional layer, followed by ReLU activation.
        self.out = self.conv2.forward(self.mid).relu()

        # Downsample the feature map using average pooling with a 4x4 kernel.
        # Flatten the resulting tensor to a shape of [batch_size x 392].
        pooled = minitorch.avgpool2d(self.out, (4, 4)).view(BATCH, 392)

        # Pass through the first linear layer with ReLU activation and apply dropout.
        tmp = self.linear1.forward(pooled).relu()
        tmp = minitorch.dropout(tmp, rate=0.25, ignore=not self.training)

        # Pass through the second linear layer to produce class logits.
        tmp = self.linear2.forward(tmp)

        # Apply log-softmax to generate log probabilities over classes.
        out = minitorch.logsoftmax(tmp, dim=1)

        return out


def make_mnist(start, stop):
    ys = []
    X = []
    for i in range(start, stop):
        y = labels[i]
        vals = [0.0] * 10
        vals[y] = 1.0
        ys.append(vals)
        X.append([[images[i][h * W + w] for w in range(W)] for h in range(H)])
    return X, ys


def default_log_fn(epoch, total_loss, correct, total, losses, model):
    print(f"Epoch {epoch} loss {total_loss} valid acc {correct}/{total}")


class ImageTrain:
    def __init__(self):
        self.model = Network()

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=BACKEND))

    def train(
        self, data_train, data_val, learning_rate, max_epochs=500, log_fn=default_log_fn
    ):
        (X_train, y_train) = data_train
        (X_val, y_val) = data_val
        self.model = Network()
        model = self.model
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            for batch_num, example_num in enumerate(
                range(0, n_training_samples, BATCH)
            ):
                if n_training_samples - example_num <= BATCH:
                    continue
                y = minitorch.tensor(
                    y_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x = minitorch.tensor(
                    X_train[example_num : example_num + BATCH], backend=BACKEND
                )
                x.requires_grad_(True)
                y.requires_grad_(True)
                # Forward
                out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                prob = (out * y).sum(1)
                loss = -(prob / y.shape[0]).sum()

                assert loss.backend == BACKEND
                loss.view(1).backward()

                total_loss += loss[0]
                losses.append(total_loss)

                # Update
                optim.step()

                if batch_num % 5 == 0:
                    model.eval()
                    # Evaluate on 5 held-out batches

                    correct = 0
                    for val_example_num in range(0, 1 * BATCH, BATCH):
                        y = minitorch.tensor(
                            y_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        x = minitorch.tensor(
                            X_val[val_example_num : val_example_num + BATCH],
                            backend=BACKEND,
                        )
                        out = model.forward(x.view(BATCH, 1, H, W)).view(BATCH, C)
                        for i in range(BATCH):
                            m = -1000
                            ind = -1
                            for j in range(C):
                                if out[i, j] > m:
                                    ind = j
                                    m = out[i, j]
                            if y[i, ind] == 1.0:
                                correct += 1
                    log_fn(epoch, total_loss, correct, BATCH, losses, model)

                    total_loss = 0.0
                    model.train()


if __name__ == "__main__":
    data_train, data_val = (make_mnist(0, 5000), make_mnist(10000, 10500))
    ImageTrain().train(data_train, data_val, learning_rate=0.01)
