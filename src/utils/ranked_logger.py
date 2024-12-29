import logging
from typing import Mapping

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    def __init__(
        self, name: str, rank_zero_only: bool = False, extra: Mapping[str, object] | None = None
    ):
        super().__init__(logging.getLogger(name), extra)
        self._rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: int | None = None, *args, **kwargs) -> None:  # type: ignore
        if not self.isEnabledFor(level):
            return

        msg, kwargs = self.process(msg, kwargs)  # type: ignore
        current_rank = self._get_current_rank()
        msg = rank_prefixed_message(str(msg), current_rank)
        if self._rank_zero_only:
            if current_rank == 0:
                self.logger.log(level, msg, *args, **kwargs)
        else:
            if rank is None:
                self.logger.log(level, msg, *args, **kwargs)
            elif current_rank == rank:
                self.logger.log(level, msg, *args, **kwargs)

    def _get_current_rank(self) -> int:
        current_rank = getattr(rank_zero_only, "rank", None)
        if current_rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        return current_rank
