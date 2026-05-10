import sys
import warnings
from pathlib import Path

SRC_PATH = str(Path(__file__).resolve().parent.parent)
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from huggingface_hub.utils import logging as hf_logging, disable_progress_bars

# Suppress Hugging Face hub warnings
hf_logging.set_verbosity_error()
disable_progress_bars()

from rich.console import Console

from utils.logger import get_logger


logger = get_logger(__name__)


def download_model(model_id: str, local_dir: str) -> None:
    """
    Download a model from HuggingFace to a local directory.
    
    Args:
        model_id: HuggingFace model identifier
        local_dir: Local directory to save the model
    """
    logger.info(f"Downloading model '{model_id}' to '{local_dir}'")
    
    local_path = Path(local_dir)
    if not local_path.is_absolute():
        from utils.config import PROJECT_ROOT
        local_path = PROJECT_ROOT / local_dir
    
    # Create the local directory if it doesn't exist
    local_path.mkdir(parents=True, exist_ok=True)
    
    console = Console()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        with console.status(f"[bold blue]Pulling {model_id}[/bold blue]", spinner="dots"):
            try:
                snapshot_download(
                    repo_id=model_id,
                    local_dir=str(local_path)
                )
                logger.info(f"Successfully downloaded '{model_id}' to '{local_path.resolve()}'")
            
            except (GatedRepoError, RepositoryNotFoundError) as e:
                error_msg = str(e)
                if "401" in error_msg or "gated" in error_msg.lower():
                    logger.error(f"Authentication required for model '{model_id}': {e}")
                    console.print(
                        f"\n[bold red]Authentication Required[/bold red]\n"
                        f"The model [bold cyan]{model_id}[/bold cyan] is restricted or requires authentication.\n"
                        f"Please authenticate using [bold]huggingface-cli login[/bold] and ensure you have "
                        f"accepted the model's terms of use on the Hugging Face website.\n"
                    )
                else:
                    logger.error(f"Repository not found or access denied '{model_id}': {e}")
                    console.print(f"\n[red]Repository not found or access denied for [bold]{model_id}[/bold]: {e}[/red]")
                sys.exit(1)
            
            except Exception as e:
                logger.error(f"Failed to download model '{model_id}': {e}")
                console.print(f"\n[red]Failed to pull {model_id}: {e}[/red]")
                sys.exit(1)
