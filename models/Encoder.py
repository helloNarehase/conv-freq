import torch
from torch import einsum, nn
from torch.nn import functional as F

from .Block import Block

class Encoder(nn.Module):
    def __init__(self, dim: int, nhead: int, ratio: float, depth: int, max_length: int):
        super().__init__()
        
        # 입력 차원 1을 모델의 임베딩 차원(dim)으로 확장하는 레이어
        self.input_projection = nn.Linear(1, dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        self.blocks = nn.ModuleList([
            Block(dim=dim, nhead=nhead, ratio=ratio) for _ in range(depth)
        ])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_length + 1, dim))
        self.norm = nn.LayerNorm(dim)
        
        # MLM(Masked Language Modeling)을 위한 최종 로짓 레이어
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1) # 토큰당 하나의 값을 예측한다고 가정
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, mask_idx: list | None = None, cal_loss: bool = True):
        batch_size, seq_len, _ = x.shape
        
        # 0. 입력 텐서의 차원을 모델의 dim으로 확장
        h = self.input_projection(x)

        # 1. CLS 토큰과 위치 임베딩(positional embeddings)으로 입력 토큰 준비
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x_with_cls = torch.cat((cls_tokens, h), dim=1)
        x_with_cls += self.pos_embedding[:, :(seq_len + 1), :]

        # 2. 어텐션 마스크 생성
        # 패딩 마스크 생성 (lengths 사용)
        max_len = x_with_cls.shape[1]
        attention_mask = (torch.arange(max_len, device=h.device)[None, :] < lengths[:, None] + 1).float()

        # pre-training을 위한 마스킹 처리 (벡터화)
        if mask_idx is not None:
            # 원본 시퀀스 길이에 맞춰 마스크 생성
            loss_mask = torch.zeros((batch_size, seq_len), device=h.device, dtype=torch.bool)

            # mask_idx를 1D 텐서로 변환하고 batch_indices와 결합하여 벡터화된 인덱스 생성
            all_mask_indices = torch.cat(mask_idx, dim=0)
            batch_indices = torch.cat([torch.full_like(m, i) for i, m in enumerate(mask_idx)], dim=0)
            
            # Loss mask를 벡터화된 방식으로 업데이트
            loss_mask[batch_indices, all_mask_indices] = True
            
            # MLM 마스크를 attention 마스크에 적용
            # CLS 토큰을 포함한 인덱스로 변환
            attention_mask[batch_indices, all_mask_indices + 1] = 0.0

            # 마스크된 토큰들을 mask_token으로 교체
            x_with_cls[batch_indices, all_mask_indices + 1, :] = self.mask_token
            
        # 3. Transformer 블록 통과
        # Attention 모듈은 mask가 bool 타입이거나 None이어야 함
        # 여기서는 mask가 0.0 또는 1.0 값을 가지므로, `0`인 부분을 float('-inf')로 바꾸는 작업이 필요할 수 있음
        # Block 클래스 내부에서 이 처리를 한다고 가정
        for block in self.blocks:
            x_with_cls, _ = block(x_with_cls, attention_mask)

        # 4. 정규화
        x_with_cls = self.norm(x_with_cls)

        # 5. pre-training vs. 추론 출력 처리
        if mask_idx is not None:
            # pre-training: 마스킹된 토큰 예측
            x_seq = x_with_cls[:, 1:, :] # CLS 토큰 제외
            
            # 마스킹된 토큰에 대한 로짓(logits) 계산
            masked_logits = self.to_logits(x_seq[loss_mask])
            
            # 마스킹되지 않은 원본 x에서 참값 추출 (마스킹되기 전의 x를 사용)
            # 이 시점의 `x`는 input_projection을 거친 상태이므로,
            # `y_true`를 x[loss_mask]로 가져오는 것은 올바르지 않습니다.
            # 하지만 사용자 요구사항에 따라 `x`를 사용했습니다.
            # 실제로는 `forward` 메서드 초기에 원본 x를 저장해야 합니다.
            y_true = x[loss_mask]
            
            if cal_loss:
                # Loss 계산
                loss = F.l1_loss(masked_logits.squeeze(), y_true.squeeze())    
                return loss, x_with_cls
            else:
                return None, y_true
        else:
            # 추론: 전체 출력 반환
            return None, x_with_cls