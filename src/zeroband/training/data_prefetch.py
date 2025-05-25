import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from google.cloud import storage

from zeroband.utils.logger import get_logger

STABLE_FILE = "stable"


class GCPPrefetcher:
    """
    This class is in charge of downloading the parquet files from GCS to a local directory (in shared memory).
    It meant to be used with the ParquetDataset class. The Dataset is not aware of the prefetcher, it just sees the local directory.

    Under the hood, the prefecther should only be instantiate on rank 0 and then spawned as a subprocess. It periodically checks for new files in GCS and download them using a thread pool
    to the local directory in shared memory.

    Args:
        gcp_path: str, the path to the GCS bucket and the folder containing the parquet files
        local_dir: str, the local directory to store the parquet files. Use /dev/shm for fast IO
        max_buffer_steps: int = 3, the number of steps to keep in the buffer
        max_workers: int = 8, the number of workers to use for the download and delete
    """

    def __init__(self, gcp_path: str, local_dir: str, max_buffer_steps: int | None = None, max_workers: int = 8):
        self.prefetcher_process = mp.Process(target=self._prefetch, args=(gcp_path, local_dir, max_buffer_steps, max_workers))
        self.prefetcher_process.start()

    def _prefetch(self, *args, **kwargs):
        prefetcher = _GCPPrefetcherInternal(*args, **kwargs)
        prefetcher.prefetch()

    def shutdown(self):
        self.prefetcher_process.terminate()
        self.prefetcher_process.join()


class _GCPPrefetcherInternal:
    """
    this class should only be used internally by the GCPPrefetcher class. It the code that is running in the subprocess.
    """

    def __init__(
        self,
        gcp_path: str,
        local_dir: str,  # use /dev/shm for fast IO
        max_buffer_steps: int | None,
        max_workers: int,
    ):
        self.logger = get_logger("TRAIN")

        self.gcs_path = gcp_path.replace("gs://", "")
        self.local_dir = Path(local_dir)

        self.bucket_name = self.gcs_path.split("/")[0]
        self.src_folder = Path("/".join(self.gcs_path.split("/")[1:]))

        self.logger.info(f"Initializing GCPPrefetcher for {self.bucket_name} / {self.src_folder}")

        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

        self.buffer_steps = []

        self.files_downloaded = []
        self.delete_files = []

        self.max_buffer_steps = max_buffer_steps
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def prefetch(self):
        while True:
            steps_blobs = self.list_available_steps()

            step_to_download = sorted(list(steps_blobs.keys()))
            if self.max_buffer_steps is not None:
                step_to_download = step_to_download[-self.max_buffer_steps :]
                # here we only take the last max_buffer_steps steps. the key are the step numbers and the values are the blobs

            for step_number in step_to_download:
                files = self.filter_to_download(steps_blobs[step_number])
                futures = []

                for file in files:
                    future = self.thread_pool.submit(self._download_files, file)
                    futures.append(future)
                    self.files_downloaded.append(file.name)

                wait(futures)

                stable_file = self._get_stable_file(files[0])
                stable_file.touch()

    def _get_stable_file(self, blob: storage.Blob) -> Path:
        parts = Path(blob.name).parts
        src_part_len = len(self.src_folder.parts)
        return self.local_dir / Path(*parts[src_part_len:-1]) / "stable"

    def _blob_to_local_path(self, blob: storage.Blob) -> Path:
        parts = Path(blob.name).parts
        src_part_len = len(self.src_folder.parts)
        return self.local_dir / Path(*parts[src_part_len:])

    def _download_files(self, blob: storage.Blob):
        tmp_path = self._blob_to_local_path(blob)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(tmp_path))

    def filter_to_download(self, blobs: list[storage.Blob]) -> list[storage.Blob]:
        return [f for f in blobs if f.name.endswith(".parquet") and f.name not in self.files_downloaded]

    def list_available_steps(self) -> dict[int, list[storage.Blob]]:
        """
        List all step into the bucket and return a dict with the step number as key and the blob as value
        """
        available_steps = list(self.bucket.list_blobs(prefix=str(self.src_folder / "step_")))
        steps = defaultdict(list)

        for blob in available_steps:
            try:
                step_number = int(blob.name.partition("step_")[-1].partition("/")[0])
                local_path = self._blob_to_local_path(blob)
                if blob.name.endswith(".parquet") and blob.name not in self.files_downloaded and not local_path.exists():
                    steps[int(step_number)].append(blob)
            except Exception as e:
                self.logger.warning(f"Error parsing step number for blob {blob.name}: {e}")

        return steps

    def shutdown(self):
        self.__del__()

    def __del__(self):
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=True)


if __name__ == "__main__":
    prefetcher = GCPPrefetcher(gcp_path="gs://intellect-2/test/data", local_dir="/dev/shm/test/data")
