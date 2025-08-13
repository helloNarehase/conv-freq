def main():
    import torch
    import random
    from models import Encoder

    model = Encoder(dim=64, nhead=4, ratio=4.0, depth=6, max_length=128)

    B    = 3
    L    = 100
    span = 7

    mask = torch.arange(0, span)

    x = torch.randn(B, L, 1)
    lengths = torch.tensor([20, 30, 80])
    mask_idx = \
            [mask + random.randint(0, l) for l in lengths]

    loss, out = model(x, lengths, mask_idx)
    print(loss)

if __name__ == "__main__":
    main()
