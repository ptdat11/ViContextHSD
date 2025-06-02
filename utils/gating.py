import torch
import torch.nn as nn
import torch.nn.functional as F

class Gate(nn.Module):
    def __init__(self, *input_dims: int, output_dim: int) -> None:
        self.n_components = len(input_dims)
        assert self.n_components >= 2, "Number of input modalities must be >= 2"

        super().__init__()
        self.context_transforms = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for input_dim in input_dims]
        )
        # 1 context
        if self.n_components == 2:
            self.gating_weight = nn.Linear(
                in_features=sum(input_dims), out_features=output_dim
            )
        # 2 or more contexts
        elif self.n_components >= 3:
            self.gating_weights = nn.ModuleList([
                nn.Linear(in_features=sum(input_dims), out_features=output_dim)
                for _ in range(self.n_components)
            ])

    def forward(
        self, 
        *inputs: torch.Tensor,
        return_gates: bool = False
    ):
        # Shape of z and h before aggregating: BxMxD

        # Construct gate z
        cat = torch.cat(inputs, dim=-1)
        if self.n_components == 2:
            z = F.sigmoid(self.gating_weight(cat))
            z = torch.stack([z, 1 - z], dim=1)
        elif self.n_modals >= 3:
            z = torch.stack([weight(cat) for weight in self.gating_weights], dim=1)
            z = F.softmax(z, dim=1)

        # Transform x into h
        h = torch.stack(
            [
                transform(context)
                for transform, context in zip(self.context_transforms, inputs)
            ],
            dim=1,
        )
        # Apply gating
        h = h * z
        # Sum aggregation
        h = h.sum(dim=1)

        if return_gates:
            return h, z
        return h