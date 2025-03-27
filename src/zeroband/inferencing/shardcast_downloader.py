import argparse
from shardcast import ClientNode
import time
import logging
from pathlib import Path
import multiprocessing as mp

POLL_INTERVAL = 5
logger = logging.getLogger(__name__)


def main(servers: list[str], output_dir: Path, versions_to_keep: int = -1, backlog_version: int = -1):
    """
    Download the latest version of the model from the servers and delete expired versions.
    Versions will be saved as v{version}.safetensors in the output directory.

    Args:
        servers: list of servers to download from
        output_dir: directory to save the downloaded files
        versions_to_keep: number of versions to keep
        backlog_version: version to attempt to get first
    """
    client = ClientNode(servers, str(output_dir))

    while True:
        # 1. Pick the version to download
        available_versions = sorted([int(x[1:]) for x in client.list_available_versions().keys()])
        if not available_versions:
            logger.warning("No versions available")
            time.sleep(POLL_INTERVAL)
            continue
        version = available_versions[-1]
        if version <= backlog_version:
            backlog_version = -1
        if backlog_version != -1:
            version = backlog_version
            backlog_version += 1
        safetensors_filepath = output_dir / f"step_{version}/model.safetensors"

        # 2. Check if the version exists
        if version not in available_versions:
            logger.warning(f"Version {version} not found")
            time.sleep(POLL_INTERVAL)
            continue
        if safetensors_filepath.exists():
            logger.info(f"Version {version} already exists")
            time.sleep(POLL_INTERVAL)
            continue

        # 3. Download the version
        logger.info(f"Downloading version {version}")
        start = time.time()
        filepath = client.download_version(f"v{version}", str(safetensors_filepath))
        logger.info(f"Downloaded in {time.time() - start} seconds")

        # 4. Make stable file if successful and delete expired versions
        if filepath is not None:
            (output_dir / f"step_{version}/stable").touch()

            if versions_to_keep != -1:
                try:
                    logger.info(f"Deleting expired version {version - versions_to_keep}")
                    expired_version = output_dir / f"step_{version - versions_to_keep}/model.safetensors"
                    expired_version.unlink()
                    (output_dir / f"step_{version - versions_to_keep}/stable").unlink()
                except FileNotFoundError:
                    logger.warning(f"Expired version {version - versions_to_keep} not found")
                except Exception as e:
                    logger.warning(f"Error deleting expired version {version - versions_to_keep}: {e}")


def run_main_bg(servers: list[str], output_dir: Path, versions_to_keep: int = -1, backlog_version: int = -1) -> mp.Process:
    """
    Run the main function in a background process.

    Args:
        servers: list of servers to download from
        output_dir: directory to save the downloaded files
        versions_to_keep: number of versions to keep
        backlog_version: version to attempt to get first

    Returns:
        mp.Process: The created process running the main function
    """
    process = mp.Process(target=main, args=(servers, output_dir, versions_to_keep, backlog_version))
    process.start()
    return process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--servers", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--versions-to-keep", type=int, default=-1)
    parser.add_argument("--backlog-version", type=int, default=-1)
    args = parser.parse_args()
    main(args.servers, Path(args.output_dir), args.versions_to_keep, args.backlog_version)
