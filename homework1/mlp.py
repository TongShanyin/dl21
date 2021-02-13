import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def activation(self, z, type):
        """
        Args:
            z: tensor shape (batch_size, linear_out_features)
            type: string: relu | sigmoid | identity
        Return:
            dzdz: element-wisely gradient tensor (batch_size, linear_out_features)
        """
        if type == 'relu': # ReLU fct and its gradient
            dzdz = torch.zeros(z.size())
            dzdz[z>=0] = 1
            z[z<0]=0
        elif type == 'sigmoid': # sigmoid fct and its gradient
            dzdz = torch.exp(-z) / (1 + torch.exp(-z)).pow(2)
            z = 1 / (1 + torch.exp(-z))
        else: # identity fct and its gradient
            dzdz = torch.ones(z.size())
        return dzdz




    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)

        Return:
            y_hat: the prediction tensor (batch_size, linear_2_out_features)
        """
        # TODO: Implement the forward function, cache internal gradients
        # linear_1
        z = torch.mm(x, (self.parameters[W1]).t()) + self.parameters[b1] # tensor (batch_size, linear_1_out_features)
        # f_function
        dzdz = self.activation(z, self.f_function) # z: changes, dzdz: gradient tensor (batch_size, linear_1_out_features)
        # linear_2
        y_hat = torch.mm(z, (self.parameters[W2]).t()) + self.parameters[b2] # tensor (batch_size, linear_2_out_features)
        # g_function
        dydz = self.activation(y_hat, self.g_function) # y_hat: changes, dydz: gradient tensor (batch_size, linear_1_out_features)

        return y_hat

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        pass


    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """

    # Implement the mse loss
    # batch_size = y.size(0)
    # linear_2_out_features = y.size(1)
    loss = sum(sum((y_hat-y).pow(2)))
    dJdy_hat = 2*(y_hat-y)
    # taking mean
    loss = loss/y.numel()
    dJdy_hat = dJdy_hat/y.numel()
    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor

    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # Implement the bce loss
    batch_size = y.size(0)
    linear_2_out_features = y.size(1)
    loss = -sum(sum(y*torch.log(y_hat)+(1-y)*torch.log(1-y_hat)))/linear_2_out_features
    dJdy_hat = -(y/y_hat-(1-y)/(1-y_hat))/linear_2_out_features
    # taking mean
    loss = loss/batch_size
    dJdy_hat = dJdy_hat/batch_size
    return loss, dJdy_hat
