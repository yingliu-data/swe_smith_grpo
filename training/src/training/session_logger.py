from __future__ import annotations

from pathlib import Path

from common.logging import TraceLogger
from common.session import SessionDir, Ticket


class RunLogger:
    """Thin facade that ties a SessionDir + its TraceLogger + ticket sequence."""

    def __init__(self, session: SessionDir):
        self._session = session
        self._trace = TraceLogger(session.trace_path)
        self._seq = 0

    @property
    def session(self) -> SessionDir:
        return self._session

    @property
    def trace(self) -> TraceLogger:
        return self._trace

    def next_ticket(self, operation: str, inputs: dict | None = None) -> Ticket:
        self._seq += 1
        from common.ids import make_ticket_id

        return Ticket.start(
            tickets_dir=self._session.tickets,
            ticket_id=make_ticket_id(self._seq, operation),
            operation=operation,
            inputs=inputs or {},
        )
