import gdown
import pathlib
import logging

logger = logging.getLogger(__name__)


def download_required_files():
    """
    Downloads the models.json & embeddings.json files from Google Drive
    """
    output_dir = pathlib.Path(__file__).parent / 'models'
    gdown.download(
        'https://drive.google.com/uc?id=1e1Af5bfiFwI-W08aOfADxRePXPeT_IEl',
        str((output_dir / 'vocab.json').resolve())
    )
    gdown.download(
        'https://drive.google.com/uc?id=1rN2WwU3WyAZvRcirjR5ypt-FUyW3gC1o',
        str((output_dir / 'embeddings.json').resolve())
    )
