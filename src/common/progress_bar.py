from typing import Optional
from tqdm.auto import tqdm


class ProgressBar:
    """Shows progress bar of a process with tqdm library."""

    def __init__(
        self,
        total: Optional[int],
        description: str,
        unit: str = "B",
        unit_scale: bool = True,
        unit_divisor: int = 1024,
    ):
        self._progress_bar = tqdm(
            total=total,
            desc=description,
            unit=unit,
            unit_scale=unit_scale,
            unit_divisor=unit_divisor,
        )

    def __enter__(self):
        return self

    def set_meta(self, meta=None, **kwargs):
        """Called by MinIO before transfers; allows adjusting progress metadata."""
        
        total = None
        object_name = None

        if meta:
            total = getattr(meta, "size", None) or getattr(meta, "length", None)
            object_name = getattr(meta, "object_name", None)

        total = kwargs.get("total", total)
        total = kwargs.get("size", total)
        total = kwargs.get("length", total)
        object_name = kwargs.get("object_name", object_name)

        if total and self._progress_bar.total is None:
            self._progress_bar.total = total

        if object_name:
            self._progress_bar.set_description(f"Uploading {object_name}", refresh=False)

    def update(self, bytes_amount: int):
        self._progress_bar.update(bytes_amount)

    def set_description(self, description: str, refresh: bool = False):
        self._progress_bar.set_description(description, refresh=refresh)

    def write(self, message: str):
        tqdm.write(message)

    def close(self):
        self._progress_bar.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
