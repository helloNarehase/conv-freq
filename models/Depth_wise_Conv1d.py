import torch
from torch import einsum, nn
from torch.nn import functional as F

class DepthwiseConv1d(nn.Module):
    """
    1D Depth-wise convolution.
    
    각 입력 채널에 대해 독립적인 컨볼루션 필터를 적용합니다.
    nn.Conv1d에서 groups 파라미터를 in_channels와 동일하게 설정하여 구현합니다.
    """
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        """
        Args:
            in_channels (int): 입력 텐서의 채널 수. 출력 채널 수도 이와 동일합니다.
            kernel_size (int): 컨볼루션 커널의 크기.
            stride (int, optional): 컨볼루션의 스트라이드. Defaults to 1.
            padding (int, optional): 입력의 양쪽에 적용될 제로 패딩의 양. Defaults to 0.
            dilation (int, optional): 커널 요소 사이의 간격. Defaults to 1.
            bias (bool, optional): 학습 가능한 편향(bias)을 추가할지 여부. Defaults to True.
        """
        super().__init__()
        # Depth-wise convolution의 핵심: out_channels와 groups를 in_channels로 설정
        self.padding = padding
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels, # 출력 채널 수는 입력 채널 수와 같음
            kernel_size=kernel_size,
            groups=in_channels,      # 그룹 수를 입력 채널 수와 같게 설정
            stride=stride,
            dilation=dilation,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 텐서. Shape: (N, C_in, L_in)
                              N: 배치 사이즈, C_in: 입력 채널 수, L_in: 시퀀스 길이

        Returns:
            torch.Tensor: 출력 텐서. Shape: (N, C_in, L_out)
        """
        h = F.pad(x, [self.padding, 0])
        return self.conv(h)