
'''
Copyright (c) 2021, Alibaba Cloud and its affiliates;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import torch
from einops import rearrange

def generate_savgol_matrix(window_length: int=5, polyorder: int=2, dtype=torch.float32) -> torch.Tensor:
    """
    Generate the Savitzky-Golay design matrix X for polynomial fitting.
    
    Args:
        window_length (int): Length of the filter window (must be odd).
        polyorder (int): Order of the polynomial fit.
        dtype (torch.dtype): Data type of the output tensor.
    
    Returns:
        torch.Tensor: Design matrix X of shape (window_length, polyorder + 1).
    """
    if window_length % 2 != 1:
        raise ValueError("window_length must be odd.")
    
    m = (window_length - 1) // 2  # Half-width of the window
    x = torch.arange(-m, m + 1, dtype=dtype)  # [-m, -m+1, ..., m]
    
    # Construct Vandermonde matrix: each row is [1, x, x^2, ..., x^K]
    X = x.unsqueeze(1).pow(torch.arange(polyorder + 1, dtype=dtype))
    
    return X

def generate_savgol_coefficients(window_length: int=5, polyorder: int=2, dtype=torch.float32):
    X = generate_savgol_matrix(window_length, polyorder, dtype)
    XT = X.transpose(0, 1)
    mat = torch.linalg.inv(XT @ X) @ XT
    return mat[0, :] # Return the first row of the inverse matrix


class Filter1D(torch.nn.Module):
    def __init__(self, window_length:int = 7, polyorder:int =2):
        super().__init__()
        self.window_length =window_length
        coefficients = generate_savgol_coefficients(window_length=window_length, polyorder=polyorder)
        self.register_buffer('coefficients', coefficients.view(1, 1, -1))
        
    def forward(self, input_tensor:torch.Tensor, format='btc'):
        x = input_tensor.clone()
        if format == 'btc':
            B, T, C = x.shape
            x = rearrange(x, 'b t c -> (b c) 1 t')
            r = torch.nn.functional.conv1d(x, self.coefficients)
            x[..., self.window_length//2:-(self.window_length//2)] = r
            x = rearrange(x, '(b c) 1 t -> b t c', b=B, c=C)
        elif format == 'bct':
            B, C, T = x.shape
            x = rearrange(x, 'b c t -> (b c) 1 t')
            r = torch.nn.functional.conv1d(x, self.coefficients)
            x[..., self.window_length//2:-(self.window_length//2)] = r
            x = rearrange(x, '(b c) 1 t -> b c t', b=B, c=C)
        else:
            raise ValueError("Unsupported format. Use 'btc' or 'bct'.")
        return x        


if __name__ == '__main__':
    coefficients = generate_savgol_coefficients(window_length=5, polyorder=2)
    r = torch.nn.functional.conv1d(torch.ones(1, 2, 10), coefficients.view(1, 1, -1).expand(1, 2, -1))
    print(r)