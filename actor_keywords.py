"""Keyword covariates for the alignment regression, read off the profile descriptions.

The actor file arrived with one metadata column, `m_grpname`, which was a verbatim copy
of `grpname` -- so the only covariate available to beta was the community label the model
is scored against, and every arm that scored community recovery had to run with an
intercept-only design to stay honest. That left the alignment channel unexercised and V7
unable to discriminate, which is what blocks any real test of the facet count.

These columns are the replacement: bag-of-keyword indicators over the profile text, each
0/1 for every actor (a blank cell would make `constraints.metadata` read the column as
categorical rather than numeric). They are deliberately not a partition of the
communities -- most cut across several -- so beta has something to generalize the pins
with that is not the answer key.

Four of them (`m_maga`, `m_resist`, `m_guns`, `m_prolife`) are near-proxies for a single
community: 89%, 88%, 85% and 100% of their hits fall in one group. They are included
because they carry the strongest ideological signal in the file, and because the
community variable itself is being dropped. Note that `group_derived_columns` does NOT
flag them -- for a binary column the 0 level spans every community, so only a column like
`m_grpname`, where every value maps to one group, is caught automatically. Whether a
community-recovery score computed against a design containing them means anything is a
judgment call, not something the code will make for you.
"""

import csv
import re
import sys
from pathlib import Path

PATTERNS = {
    "m_media":        r"journalist|reporter|correspondent|editor|columnist|anchor|"
                      r"newsroom|\bpress\b|contributor @|staff writer|bureau|@nytimes|"
                      r"@washingtonpost|@cnn|@msnbc|@foxnews|@politico|@axios|@reuters|"
                      r"@npr",
    "m_family":       r"\bmom\b|\bdad\b|father|mother|husband|\bwife\b|grandma|grandpa|"
                      r"nana|married|daughter|parent",
    "m_faith":        r"\bgod\b|christian|\bfaith\b|jesus|blessed|\bchurch\b|catholic|"
                      r"believer|\bpastor\b|scripture",
    "m_pronouns":     r"she/her|he/him|they/them|she/they|he/they|\(pronouns",
    # `resistance` alone caught "antibiotic resistance"; the self-identification forms do
    # not have that problem.
    "m_resist":       r"#resist\b|#theresistance|the resistance|\bresister\b|"
                      r"#impeach|impeach45|impeachtrump|bluewave|votebluenomatterwho|"
                      r"#fbr|nevertrump",
    # `abolish|defund` named a different politics (#AbolishICE, "Abolish the Value Form")
    # and is left out rather than folded in here.
    "m_blm":          r"\bblm\b|black lives|blacklivesmatter",
    # Singular only, and it is a rule about usage rather than a patch for one bio: the
    # plural is how people name their opponents, the singular is how they name themselves.
    # Every plural in this file is oppositional -- "Republicans are ... a group of thugs",
    # "Imagining Life w/o Liberals", "I destroy liberals", "Black People Don't Have to be
    # Democrats" -- while every self-identification is singular ("Proud Democrat",
    # "Lifelong Democrat", "Conservative Christian"). `democratic` is kept because the
    # self-descriptions use it ("Democratic Candidate for US Congress").
    "m_conservative": r"\bconservative\b|\brepublican\b|\bgop\b|\blibertarian\b|"
                      r"\bconstitutionalist\b",
    "m_liberal":      r"\bliberal\b|\bprogressive\b|\bdemocrat\b|\bdemocratic\b|"
                      r"\bleftist\b|\bsocialist\b|\bdsa\b",
    # the four near-proxies
    "m_maga":         r"#maga\b|#kag|maga\b|trump2020|americafirst|wwg1wga|trumptrain|"
                      r"draintheswamp|patriot\b|\bkag\b|qanon|qarmy|potus45",
    "m_guns":         r"#2a\b|\b2a\b|\bnra\b|second amendment|gun rights|concealed",
    "m_prolife":      r"prolife|pro-life|pro life",
}

# Keywords carry no stance, and the words for a politics are also the words its opponents
# use to name it. Known cases in this file, all left in: "Black People Don't Have to be
# Democrats. #BLEXIT" and three MAGA bios ("rather be a black redneck than a white
# liberal", "GULAG EVERY LIBERAL", "Imagining Life w/o Liberals") score m_liberal; three
# Big Tent Dems bios ("Republicans are ... a group of thugs", "Ex-GOP", "Republicans are
# in the Luxury business of cultural poverty") score m_conservative; "Military Anchor
# Baby" scores m_media. The errors are not symmetric -- each lands on an actor of the
# opposing side -- so m_liberal and m_conservative carry a known bias against the
# direction beta reads them. Coding the descriptions through run.py's source/stance
# prompt is what would fix it properly.

DROP = ["m_grpname"]   # a verbatim copy of grpname; see the module docstring


def annotate(path: Path, drop=DROP) -> tuple[list[dict], list[str]]:
    rows = list(csv.DictReader(path.open()))
    cols = [c for c in rows[0] if c and c not in drop]
    for name, pat in PATTERNS.items():
        rx = re.compile(pat, re.I)
        for r in rows:
            r[name] = int(bool(rx.search(r.get("description", "") or "")))
    return rows, cols + list(PATTERNS)


def write(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    for p in (Path("actor_ideology_affiliation.csv"), Path("snapshot/actors.csv")):
        rows, cols = annotate(p)
        write(p, rows, cols)
        n = {k: sum(r[k] for r in rows) for k in PATTERNS}
        print(f"{p}: {len(rows)} rows, dropped {DROP}, added {len(PATTERNS)} columns")
        print("   " + "  ".join(f"{k.replace('m_', '')}:{v}" for k, v in n.items()))
