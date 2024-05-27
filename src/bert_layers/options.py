from .normalization import NORM2CLS
from .embeddings import EBB2CLS
from .activation import ACT2CLS
from .attention import ATTN2CLS
from .mlp import MLP2CLS
from .layers import LAYER2CLS


def print_layer_options():
    print("Activation options:")
    for option in ACT2CLS:
        print(f"  {option}")

    print("\nAttention Layer options:")
    for option in ATTN2CLS:
        print(f"  {option}")

    print("\nEmbedding Layer options:")
    for option in EBB2CLS:
        print(f"  {option}")

    print("\nBert Layer options:")
    for option in LAYER2CLS:
        print(f"  {option}")

    print("\nMLP Layer options:")
    for option in MLP2CLS:
        print(f"  {option}")

    print("\nNormalization options:")
    for option in NORM2CLS:
        print(f"  {option}")
