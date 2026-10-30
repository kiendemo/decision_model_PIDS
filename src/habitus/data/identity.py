"""Process identity of a subject record: executable path, file name, normalised command line and the
creating parent's name. The card builder turns it into a process card's invariant section.

Vendored verbatim from the original reflex/identity.py (image_dir dropped: unused).
"""

import re
from dataclasses import dataclass

from habitus.data.resolver import basename

# OpTC names images by device path; the command line on the same record says C:/... . Mapping the
# former onto the latter is a fixed rewrite of the collector's convention, not something an
# attacker influences, and it lets one model read every Windows collector the same way.
_DEVICE = re.compile(r"^/device/harddiskvolume\d+/", re.IGNORECASE)
_GUID = re.compile(r"\{?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}?")
_LONG_NUM = re.compile(r"\d{6,}")
MAX_CMD = 300


def _clean_path(path):
    if not path:
        return None
    p = path.strip().strip('"')
    if p.endswith("(deleted)"):
        # Deletion is its own invariant signal (`image_deleted`); the model judges the name.
        p = p[: -len("(deleted)")].rstrip()
    if _DEVICE.match(p):
        p = "C:/" + _DEVICE.sub("", p)
    return p


def normalize_cmd(cmd):
    """Collapse per-instance noise (GUIDs, pids, timestamps) so one program invocation pattern
    is one identity, and bound the length the model reads."""
    if not cmd:
        return ""
    c = _GUID.sub("<guid>", cmd.strip())
    c = _LONG_NUM.sub("<n>", c)
    return c[:MAX_CMD]


@dataclass(frozen=True)
class ProcessIdentity:
    os: str
    image: str          # executable path, "" when the collector did not record one (CADETS)
    name: str           # executable file name, or the process name when there is no path
    cmdline: str        # normalized command line, "" when absent or identical to the name
    parent: str = ""    # file name of the process that created it ("" when not recorded)

    @property
    def key(self):
        base = f"{self.os}\x1f{self.image}\x1f{self.name}\x1f{self.cmdline}"
        return base + (f"\x1f{self.parent}" if self.parent else "")

    def state(self):
        """The identity fields as a dict (the card's invariant section is built from it)."""
        s = {"os": self.os, "file_name": self.name}
        if self.image:          # CADETS records no image path: those questions are not asked
            s["executable"] = self.image
        if self.cmdline:
            s["command_line"] = self.cmdline
        if self.parent:         # lineage: the unique creation record, not a neighbour
            s["parent_name"] = self.parent
        return s


def process_identity(os_family, path, cmd, parent=None, exec_path=None):
    """ProcessIdentity for a subject record (path, cmd) from the resolver cache; `parent` is the
    creating process's own (path, cmd) record, when the collector recorded the creation.
    `exec_path` is the file the graph shows this process being executed from; it stands in for
    the image when the collector recorded none (CADETS stores NULL there) — the same fallback
    `Window.exec_source` uses, in the other direction."""
    image = _clean_path(path) or _clean_path(exec_path) or ""
    cmd_n = normalize_cmd(cmd)
    name = basename(image) if image else (cmd_n.split()[0].strip('"') if cmd_n else "")
    name = basename(name) or name
    if cmd_n and cmd_n.strip('"') in (name, image):
        cmd_n = ""          # a bare name repeats the image; it adds nothing to read
    pname = ""
    if parent is not None:
        pimg = _clean_path(parent[0]) or ""
        pcmd = normalize_cmd(parent[1])
        pname = basename(pimg) if pimg else (pcmd.split()[0].strip('"') if pcmd else "")
        pname = basename(pname) or pname or ""
    return ProcessIdentity(os=os_family, image=image, name=name or "", cmdline=cmd_n,
                           parent=pname)
