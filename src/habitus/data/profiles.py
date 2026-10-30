"""Per-collector-family conventions used by the card builder: the operation strings that mean "a
process image started" and "the actor changed this object", and the world-writable staging
directories of the operating system.

Vendored verbatim from the original defense/profiles.py (the dataset table lists the in-scope hosts
only; DARPA TC stays the default).
"""

from dataclasses import dataclass


def normalize_path(path):
    """Lower-cased, forward-slash form used for directory comparisons.

    Windows telemetry mixes `C:\\Users\\...` (Carbon Black) with
    `/Device/HarddiskVolume1/Users/...` (OpTC) and is case-insensitive; POSIX paths are
    unaffected by either rule in practice.
    """
    if not path:
        return None
    return path.replace("\\", "/").lower()


# Staging areas an unprivileged attacker can write to and execute from.
POSIX_STAGING = ("/tmp/", "/dev/shm/", "/var/tmp/", "/run/shm/")
WINDOWS_STAGING = (
    "/appdata/local/temp/", "/appdata/roaming/", "/windows/temp/",
    "/users/public/", "/programdata/", "/$recycle.bin/", "/perflogs/",
)


@dataclass(frozen=True)
class HostProfile:
    name: str
    execute: frozenset      # ops meaning "a process image started"
    write: frozenset        # ops meaning "the actor modified the object"
    staging_dirs: tuple
    # Does the collector record an EXECUTE edge to the *file* that was run? OpTC and Carbon
    # Black report process starts without naming the image as an object, so the exec source
    # can only come from the process record itself.
    execute_names_image: bool = True


DARPA_TC = HostProfile(
    "darpa-tc", frozenset({"EVENT_EXECUTE"}), frozenset({"EVENT_WRITE"}), POSIX_STAGING)

OPTC = HostProfile(
    "optc", frozenset({"START"}),
    frozenset({"WRITE", "MODIFY", "CREATE", "RENAME"}),
    WINDOWS_STAGING, execute_names_image=False)

CARBON_BLACK_EDR = HostProfile(
    "carbon-black-edr", frozenset({"PROCESS_EXECUTED"}),
    frozenset({"FILE_WRITTEN", "FILE_CREATED", "FILE_RENAMED", "FILE_TRUNCATED",
               "FILE_COPIED", "FILE_DELETED"}),
    WINDOWS_STAGING, execute_names_image=False)

_BY_DATASET = {
    "optc_h501": OPTC,
    "atlasv2_edr": CARBON_BLACK_EDR,
}


def profile_for(dataset_name):
    """The host profile for a dataset; DARPA TC (E3) is the default."""
    return _BY_DATASET.get(str(dataset_name).lower(), DARPA_TC)


def in_staging_dir(path, profile=DARPA_TC):
    p = normalize_path(path)
    return bool(p) and any(d in p for d in profile.staging_dirs)
