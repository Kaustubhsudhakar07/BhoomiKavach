"""
BHOOmiKAVACH AI — Background Backend Service Manager
====================================================
Maintained for backwards-compatibility.
In-process direct execution is handled seamlessly by BhoomiApiClient.
"""


def auto_start_local_backend() -> bool:
    """Instantaneous no-op: Direct in-process execution is active."""
    return True
