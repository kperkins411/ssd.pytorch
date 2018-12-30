

def do_requires_grad(model,*,requires_grad, apply_to_this_layer_on =0):
    '''
    Sets a model to be learnable or not
    :param model: network
    :param requires_grad:True gradient will be calculated, weights updated
    :param apply_to_this_layer_on will apply above to that layer forward
    :return:
    '''
    if model is None:
        raise ValueError("do_requires_grad() called with model==None")

    for i,param in enumerate(model.parameters()):
        if (i>=apply_to_this_layer_on):
            param.requires_grad = requires_grad
