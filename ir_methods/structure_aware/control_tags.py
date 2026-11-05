"""Flat control tags (scored rows of the full2 study) for every full3 tag. `python control_tags.py --tags` lists all control tags, and the staging step copies their scores.
control(tag) -> list of control tags, first is the primary paired control."""
import re, sys
RULES = [
    (r"^f3_probe_", ["f2_z2_ftpub"]),
    (r"^f3_sect06_", ["f2_z2_ftpub"]),
    (r"^f3_sp06_.*_s1$", ["f2_ftpub_s1"]),
    (r"^f3_sp06_.*_s2$", ["f2_ftpub_s2"]),
    (r"^f3_sp06_.*_s3$", ["f2_ftpub_s3"]),
    (r"^f3_sp06_", ["f2_ftpub_s1", "f2_z2_ftpub"]),
    (r"^f3_sect4b_.*_exp", ["f2_ftpub4b2x_exp_fuse0_k200_m25", "f2_ftpub4b2x_fuse0_k200"]),
    (r"^f3_sect4b_", ["f2_ftpub4b2x_fuse0_k200", "f2_ftpub4b2x_exp_fuse0_k200_m25"]),
    (r"^f3_sp4b_.*_exp", ["f2_ftpub4b_exp_fuse0_k200_m25", "f2_ftpub4b_fuse0_k200"]),
    (r"^f3_sp4b_", ["f2_ftpub4b_fuse0_k200", "f2_ftpub4b_exp_fuse0_k200_m25"]),
    (r"^f3_toc", ["f2_ftpub_s1", "f2_z2_ftpub"]),
]
ALL = sorted({t for _, ts in RULES for t in ts} | {"f2_ftpub4b2x_exp_fuse0_k200_m25", "f2_z2_zs06", "f2_z2_first"})
def control(tag):
    for pat, ts in RULES:
        if re.search(pat, tag):
            return ts
    return []
if __name__ == "__main__":
    if "--tags" in sys.argv:
        print(" ".join(ALL))
    else:
        for t in sys.argv[1:]:
            print(t, control(t))
