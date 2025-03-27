import queue
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Callable
from google.cloud import storage

from zeroband.logger import get_logger
import multiprocessing as mp


class PrioritizedJob:
    """
    just a wrapper to make the priority queue work
    """

    def __init__(self, priority: int, job: Callable, args: tuple = tuple(), kwargs: dict = dict()):
        self.priority = priority
        self.job = job
        self.args = args
        self.kwargs = kwargs

    def __lt__(self, other):
        return self.priority < other.priority

    def __eq__(self, other):
        return self.priority == other.priority

    def execute(self):
        self.job(*self.args, **self.kwargs)


class PriorityThreadPool:
    """

    this class is allow to treat job asynchroneously with a priority.

    Under the hood it used a thread pool and a priority queue to treat the job.
    """

    def __init__(self, max_workers):
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.priority_queue = queue.PriorityQueue()

        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.daemon = True  # allow to avoid blocking the main thread when shutting down

        self.running_tag = False

        self.start()

    def start(self):
        self.running_tag = True
        self.worker_thread.start()

    def stop(self):
        self.running_tag = False
        self.worker_thread.join()

    def _worker_loop(self):
        while self.running_tag:
            try:
                prioritized_job = self.priority_queue.get()
                self.thread_pool.submit(prioritized_job.execute)
            except queue.Empty:
                continue

    def submit(self, func, *args, priority: int = 0, **kwargs):
        self.priority_queue.put(PrioritizedJob(priority, func, args, kwargs))

    def shutdown(self, wait=True):
        self.running_tag = False
        if wait:
            self.worker_thread.join()
        else:
            self.worker_thread.join(timeout=1)


class GCPPrefetcher:
    """
    This class is in charge of downloading the parquet files from GCS to a local directory (in shared memory).
    It meant to be used with the ParquetDataset class. The Dataset is not aware of the prefetcher, it just sees the local directory.

    Under the hood, the prefecther should only be instantiate on rank 0 and then spawed as a subprocess. It periodically checks for new files in GCS and download them using a thread pool
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
        self.logger = get_logger()

        self.gcs_path = gcp_path.replace("gs://", "")
        self.local_dir = Path(local_dir)

        self.bucket_name = self.gcs_path.split("/")[0]
        self.src_folder = Path("/".join(self.gcs_path.split("/")[1:]))

        self.logger.info(f"Initializing GCPPrefetcher for {self.bucket_name} / {self.src_folder}")

        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

        self.thread_pool_download = PriorityThreadPool(max_workers=max_workers)
        self.thread_pool_delete = PriorityThreadPool(max_workers=max_workers)

        self.buffer_steps = []

        self.files_downloaded = []
        self.delete_files = []

        self.max_buffer_steps = max_buffer_steps

    def prefetch(self):
        while True:
            steps_blobs = self.list_available_steps()

            step_to_download = sorted(list(steps_blobs.keys()))
            if self.max_buffer_steps is not None:
                step_to_download = step_to_download[-self.max_buffer_steps :]
                # here we only take the last max_buffer_steps steps. the key are the step numbers and the values are the blobs

            for step_number in step_to_download:
                files = self.filter_to_download(steps_blobs[step_number])
                for file in files:
                    self.thread_pool_download.submit(self._download_files, file, priority=step_number)
                    # we want to download the file from the oldest step first
                    self.files_downloaded.append(file.name)

            # TODO bring back deleting logic
            # the issue is that right now there is no way to know if the file is still in use during training
            # technically this should be not a problem with a large enough buffer
            # but for testing removing the delete logic is easier
            # its not trivial neither to make the dataset and the prefetcher because both work in different process and the dataloader is in wrapping datasets.

            # step_to_delete = list(steps_blobs.keys())[: -self.max_buffer_steps]

            # for step_number in step_to_delete:
            #     files = steps_blobs[step_number]
            #     for file in files:
            #         self.delete_files.append(file.name)
            #         # we want to delete the file from the oldest step first
            #         self.thread_pool_delete.submit(step_number, self._delete_files, file)

    def _blob_to_local_path(self, blob: storage.Blob) -> Path:
        parts = Path(blob.name).parts
        src_part_len = len(self.src_folder.parts)
        return self.local_dir / Path(*parts[src_part_len:])

    def _delete_files(self, blob: storage.Blob):
        local_path = self._blob_to_local_path(blob)
        if local_path.exists():
            local_path.unlink()

    def _download_files(self, blob: storage.Blob):
        local_path = self._blob_to_local_path(blob)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

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
                if blob.name.endswith(".parquet") and blob.name not in self.files_downloaded:
                    steps[int(step_number)].append(blob)
            except Exception as e:
                self.logger.warning(f"Error parsing step number for blob {blob.name}: {e}")

        return steps

    def shutdown(self):
        self.__del__()

    def __del__(self):
        if hasattr(self, "thread_pool_download"):
            self.thread_pool_download.shutdown(wait=True)
        if hasattr(self, "thread_pool_delete"):
            self.thread_pool_delete.shutdown(wait=True)


if __name__ == "__main__":
    prefetcher = GCPPrefetcher(gcp_path="gs://intellect-2/test/data", local_dir="/dev/shm/test/data")
