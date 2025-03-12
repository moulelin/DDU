import torch
from torch.nn.functional import normalize, conv2d, conv_transpose2d


class SpectralNormConv(object):
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 1

    # Version 1 changes:
    #   - Made `W` not a buffer.
    #   - Added `v` as a buffer.
    #   - Made eval mode use W = u @ W_orig @ v rather than the stored W.

    def __init__(self, coeff, min_coeff, input_dim, name="weight", n_power_iterations=1, eps=1e-12):
        """
        Args:
            coeff (float): Upper bound for the spectral norm (largest singular value).
            min_coeff (float): Lower bound for the minimal singular value.
            input_dim (tuple or torch.Size): Shape of the input (e.g., [in_channels, height, width]).
            name (str): Name of the weight parameter.
            n_power_iterations (int): Number of power iterations to approximate the top singular value.
            eps (float): Small constant for numerical stability.
        """
        self.coeff = coeff
        self.min_coeff = min_coeff
        self.input_dim = input_dim  # expected as a 3D shape: (in_channels, H, W)
        self.name = name
        if n_power_iterations <= 0:
            raise ValueError("Expected n_power_iterations to be positive, but got {}".format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module, do_power_iteration):
        # Retrieve the original weight and the power iteration buffers.
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        sigma_log = getattr(module, self.name + "_sigma")  # for logging

        # Get convolution settings from the module.
        stride = module.stride
        padding = module.padding

        # Update the power iteration vectors if needed.
        if do_power_iteration:
            with torch.no_grad():
                output_padding = 0
                if stride[0] > 1:
                    # Note: this may not generalize to strides > 2.
                    output_padding = 1 - self.input_dim[-1] % 2
                for _ in range(self.n_power_iterations):
                    # Update v: apply the transposed convolution.
                    v_s = conv_transpose2d(
                        u.view(self.out_shape),
                        weight,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                    )
                    # In-place normalization of v.
                    v = normalize(v_s.view(-1), dim=0, eps=self.eps, out=v)

                    # Update u: apply the convolution.
                    u_s = conv2d(v.view(self.input_dim), weight, stride=stride, padding=padding, bias=None)
                    u = normalize(u_s.view(-1), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        # Compute the approximate maximum singular value (sigma) using the current u and v.
        weight_v = conv2d(v.view(self.input_dim), weight, stride=stride, padding=padding, bias=None)
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)

        # Enforce the upper bound on the spectral norm.
        factorReverse = torch.max(torch.ones(1, device=weight.device), sigma / self.coeff)
        weight = weight / (factorReverse + 1e-5)

        # For logging: store the unnormalized sigma.
        sigma_log.copy_(sigma.detach())

        # --- New part: Enforce a lower bound on the minimal singular value ---
        # Reshape the weight tensor to a 2D matrix.
        # For a convolution kernel of shape (out_channels, in_channels, k_h, k_w),
        # we reshape it to (out_channels, in_channels * k_h * k_w).
        weight_mat = weight.view(weight.size(0), -1)
        # Compute the full singular value decomposition.
        try:
            # Use torch.linalg.svd if available (PyTorch 1.8+).
            U, S, Vh = torch.linalg.svd(weight_mat, full_matrices=False)
        except AttributeError:
            # Fallback to torch.svd for older versions.
            U, S, V = torch.svd(weight_mat)
        sigma_min = S.min()
        # If the minimal singular value is below the desired lower bound, scale up the weights.
        if sigma_min < self.min_coeff:
            lower_factor = self.min_coeff / (sigma_min + self.eps)
            weight = weight * lower_factor
        # ----------------------------------------------------------------------

        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        # Check that the input dimensions match.
        assert inputs[0].shape[1:] == self.input_dim[1:], "Input dims don't match actual input"
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    @staticmethod
    def apply(module, coeff, min_coeff, input_dim, name, n_power_iterations, eps):
        # Prevent multiple spectral norm hooks on the same parameter.
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNormConv) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on the same parameter {}".format(name))

        fn = SpectralNormConv(coeff, min_coeff, input_dim, name, n_power_iterations, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            # Initialize v: a random vector with the proper number of elements.
            num_input_dim = 1
            for d in input_dim:
                num_input_dim *= d
            v = normalize(torch.randn(num_input_dim, device=weight.device), dim=0, eps=fn.eps)

            # Get convolution settings.
            stride = module.stride
            padding = module.padding
            # A forward call to infer the output shape.
            u_init = conv2d(v.view(input_dim), weight, stride=stride, padding=padding, bias=None)
            fn.out_shape = u_init.shape
            num_output_dim = 1
            for d in fn.out_shape:
                num_output_dim *= d
            u = normalize(torch.randn(num_output_dim, device=weight.device), dim=0, eps=fn.eps)

        # Remove the original weight parameter and register it under a new name.
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)

        # Register the buffers for u, v, and sigma.
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1, device=weight.device))

        # Register the forward pre-hook to recompute the normalized weight.
        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormConvStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormConvLoadStateDictPreHook(fn))
        return fn


class SpectralNormConvLoadStateDictPreHook(object):
    # See docstring of SpectralNormConv._version regarding changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        version = local_metadata.get("spectral_norm_conv", {}).get(fn.name + ".version", None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + fn.name + "_orig"]
                weight = state_dict.pop(prefix + fn.name)
                sigma = (weight_orig / weight).mean()
                weight_mat = weight_orig.view(weight_orig.size(0), -1)
                # (The following is only for backward compatibility and logging.)
                u = state_dict[prefix + fn.name + "_u"]


class SpectralNormConvStateDictHook(object):
    # See docstring of SpectralNormConv._version regarding changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if "spectral_norm_conv" not in local_metadata:
            local_metadata["spectral_norm_conv"] = {}
        key = self.fn.name + ".version"
        if key in local_metadata["spectral_norm_conv"]:
            raise RuntimeError("Unexpected key in metadata['spectral_norm_conv']: {}".format(key))
        local_metadata["spectral_norm_conv"][key] = self.fn._version


def spectral_norm_conv(module, coeff, min_coeff, input_dim, n_power_iterations, name="weight", eps=1e-12):
    r"""Applies soft spectral normalization with both an upper and a lower bound
    on the Lipschitz constant to a convolution layer.

    Upper bound (largest singular value) is enforced via power iteration,
    while the lower bound (smallest singular value) is enforced by computing
    the full SVD on the reshaped kernel and scaling if necessary.

    Args:
        module (nn.Module): The convolution module.
        coeff (float): Upper bound for the spectral norm.
        min_coeff (float): Lower bound for the minimal singular value.
        input_dim (tuple): The shape of the input (in_channels, H, W).
        n_power_iterations (int): Number of power iterations.
        name (str): Name of the weight parameter.
        eps (float): Epsilon for numerical stability.

    Returns:
        The module with spectral normalization applied.
    """
    input_dim_4d = torch.Size([1, input_dim[0], input_dim[1], input_dim[2]])
    SpectralNormConv.apply(module, coeff, min_coeff, input_dim_4d, name, n_power_iterations, eps)
    return module


def remove_spectral_norm_conv(module, name="weight"):
    r"""Removes the spectral normalization reparameterization from a module.

    Args:
        module (nn.Module): The module containing the parameter.
        name (str): Name of the weight parameter.

    Returns:
        The module with spectral normalization removed.
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNormConv) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))
