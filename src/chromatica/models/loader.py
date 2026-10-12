"""Named architecture registry with explicit historical aliases."""

from chromatica.models.base import BaseCNN
from chromatica.models.encoder_decoder import EncoderDecoder
from chromatica.models.unet import UNet

ARCHITECTURES: dict[str, type[BaseCNN]] = {
    "encoder_decoder": EncoderDecoder,
    "unet": UNet,
}
ALIASES: dict[str, str] = {
    "v1": "encoder_decoder",
    "u_net_v1": "encoder_decoder",
    "U_NET_WITHOUT_SKIP_CONNECTIONS": "encoder_decoder",
    "v2": "unet",
    "u_net_v2": "unet",
    "U_NET_WITH_SKIP_CONNECTIONS": "unet",
}


def architecture_name(name: str) -> str:
    """Resolve a canonical name or a known historical alias."""
    canonical = ALIASES.get(name, name)
    if canonical not in ARCHITECTURES:
        msg = f"Unknown architecture {name!r}; choose {sorted(ARCHITECTURES)}."
        raise ValueError(msg)
    return canonical


def load_cnn_class(name: str) -> type[BaseCNN]:
    """Return the class registered for the requested architecture."""
    return ARCHITECTURES[architecture_name(name)]


def create_model(name: str) -> BaseCNN:
    """Construct a fresh model without loading or fetching weights."""
    return load_cnn_class(name)()
