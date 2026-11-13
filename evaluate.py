"""Score your outputs against the expected answers.

    python3 evaluate.py out expected

Compares out/<id>.json with expected/<id>.json for every expected file.
Standard library only (Python 3.9+). We use this same script on the hidden contracts.

How values are compared
- rates: matched on (start, end, room, meal_plan, occupancy); the price must be equal.
  Room names are compared loosely: case, punctuation and the word "room" are ignored,
  so "DELUXE ROOM" matches "Deluxe".
- contract fields: equal values; the hotel name may be shorter or longer
  ("Saffron Courtyard" matches "Hotel Saffron Courtyard").
- policies: each policy must match as a whole (every tier or instalment).
- a value the expected file leaves out (because it is asked as a question)
  must be left out of yours too.
- sources: for each correct value, the file and page must match; the quote must not be
  empty, and for rates it must contain the price (e.g. "4,500" or "4500"). Any quote
  length is fine. We read a sample of quotes ourselves; this script does not judge wording.
- questions: matched by their "about" field. Missed and extra questions both count.
- not_loaded: an expected item counts as found when one of yours cites the same file and page.
  Extra not_loaded items cost nothing.
- for each kind of miss the script prints up to 3 examples, not every one.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

CONTRACT_FIELDS = ["hotel", "valid_from", "valid_to", "currency", "taxes_included"]


def norm_room(s):
    words = re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()
    return " ".join(w for w in words if w not in ("room", "rooms"))


def norm_text(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())


def same_source(a, b):
    try:
        return Path(a["file"]).name == Path(b["file"]).name and int(a["page"]) == int(b["page"]) \
            and str(a.get("quote", "")).strip() != ""
    except (KeyError, TypeError, ValueError):
        return False


def price_in_quote(price, quote):
    q = str(quote).replace(",", "")
    p = f"{price:.2f}".rstrip("0").rstrip(".")
    return re.search(rf"(?<![\d.]){re.escape(p)}(?:\.0+)?(?!\d)", q) is not None


def field_equal(name, got, want):
    if name == "hotel":
        g, w = norm_text(got), norm_text(want)
        return bool(g) and (g in w or w in g)
    if name == "currency":
        return str(got).strip().upper() == str(want).strip().upper()
    return got == want


def structure_problems(out):
    probs = []
    for key in ("contract", "rates", "policies", "questions", "not_loaded"):
        if key not in out:
            probs.append(f"missing top-level key '{key}'")
    for i, r in enumerate(out.get("rates", []) if isinstance(out.get("rates"), list) else []):
        for k in ("start", "end", "room", "meal_plan", "occupancy", "price", "source"):
            if k not in r:
                probs.append(f"rates[{i}] has no '{k}'")
                break
    return probs


def score(out, exp):
    r = {"problems": structure_problems(out), "notes": []}
    src_ok = src_total = 0

    # contract
    oc, ec = out.get("contract") or {}, exp["contract"]
    ok = 0
    for f in CONTRACT_FIELDS:
        want, got = ec.get(f), oc.get(f)
        if want is None:
            if got is None:
                ok += 1
            else:
                r["notes"].append(f"contract.{f}: loaded, but the documents disagree, so it should be asked")
        elif got is None:
            r["notes"].append(f"contract.{f}: missing")
        elif field_equal(f, got.get("value"), want["value"]):
            ok += 1
            src_total += 1
            if same_source(got.get("source") or {}, want["source"]):
                src_ok += 1
            else:
                r["notes"].append(f"contract.{f}: right value, wrong source")
        else:
            r["notes"].append(f"contract.{f}: got {got.get('value')!r}, want {want['value']!r}")
    r["contract"] = (ok, len(CONTRACT_FIELDS))

    # rates
    def key(x):
        return (x.get("start"), x.get("end"), norm_room(x.get("room")), x.get("meal_plan"), x.get("occupancy"))
    want = {key(x): x for x in exp["rates"]}
    got, dupes = {}, 0
    for x in out.get("rates") or []:
        k = key(x)
        dupes += k in got
        got[k] = x
    correct = wrong = 0
    for k, w in want.items():
        g = got.get(k)
        if g is None:
            continue
        try:
            same = abs(float(g.get("price")) - w["price"]) < 0.5
        except (TypeError, ValueError):
            same = False
        if same:
            correct += 1
            src_total += 1
            s = g.get("source") or {}
            if same_source(s, w["source"]) and price_in_quote(w["price"], s.get("quote", "")):
                src_ok += 1
        else:
            wrong += 1
            if wrong <= 3:
                r["notes"].append(f"rate {k}: price {g.get('price')}, want {w['price']}")
    missing = sum(1 for k in want if k not in got)
    extra = sum(1 for k in got if k not in want) + dupes
    r["rates"] = {"expected": len(want), "correct": correct, "wrong_price": wrong, "missing": missing, "extra": extra}
    if missing:
        sample = [k for k in want if k not in got][:3]
        r["notes"].append(f"{missing} rates missing, e.g. {sample}")
    if extra:
        sample = [k for k in got if k not in want][:3]
        r["notes"].append(f"{extra} rates not expected, e.g. {sample}")

    # policies
    def tiers(items):
        return Counter((int(i.get("within_days", -1)), float(i.get("charge_percent", -1))) for i in items or [])

    def instalments(items):
        return Counter((float(i.get("percent", -1)), i.get("when"),
                        None if i.get("days_before_arrival") is None else int(i["days_before_arrival"]))
                       for i in items or [])
    op, ep = out.get("policies") or {}, exp["policies"]
    pol_ok = 0
    for name, fn in (("cancellation", tiers), ("advance_payment", instalments)):
        w, g = ep.get(name), op.get(name)
        if w is None:
            if not g:
                pol_ok += 1
            else:
                r["notes"].append(f"policies.{name}: loaded, but the documents disagree, so it should be asked")
            continue
        try:
            match = g is not None and fn(g) == fn(w)
        except (TypeError, ValueError):
            match = False
        if match:
            pol_ok += 1
            for wi in w:
                src_total += 1
                if any(same_source(gi.get("source") or {}, wi["source"]) for gi in g):
                    src_ok += 1
        else:
            r["notes"].append(f"policies.{name}: does not match")
    r["policies"] = (pol_ok, 2)

    # questions
    wq = Counter(q["about"] for q in exp["questions"])
    gq = Counter(q.get("about") for q in out.get("questions") or [])
    matched = sum((wq & gq).values())
    r["questions"] = {"expected": sum(wq.values()), "matched": matched,
                      "missed": sum((wq - gq).values()), "extra": sum((gq - wq).values())}
    if wq - gq:
        r["notes"].append(f"questions missed: {sorted((wq - gq).elements())}")
    if gq - wq:
        r["notes"].append(f"questions not needed: {sorted((gq - wq).elements())}")

    # not loaded
    found = 0
    for n in exp["not_loaded"]:
        if any(same_source(m.get("source") or {}, n["source"]) for m in out.get("not_loaded") or []):
            found += 1
        else:
            r["notes"].append(f"not_loaded missing: {n['text'][:60]!r}")
    r["not_loaded"] = (found, len(exp["not_loaded"]))
    r["sources"] = (src_ok, src_total)
    return r


def pct(a, b):
    return "100%" if b == 0 else f"{100 * a / b:.0f}%"


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "out")
    exp_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "expected")
    files = sorted(exp_dir.glob("*.json"))
    if not files:
        sys.exit(f"no expected files in {exp_dir}")
    tot = Counter()
    print(f"{'contract':<9}{'rates':>14}{'details':>9}{'policies':>10}{'sources':>9}{'questions':>11}{'not loaded':>12}")
    for f in files:
        exp = json.loads(f.read_text())
        of = out_dir / f.name
        if not of.exists():
            print(f"{f.stem:<9}  no output file ({of})")
            out = {}
        else:
            try:
                out = json.loads(of.read_text())
            except json.JSONDecodeError as e:
                print(f"{f.stem:<9}  invalid JSON: {e}")
                out = {}
        r = score(out, exp)
        ra, q = r["rates"], r["questions"]
        rate_txt = f"{ra['correct']}/{ra['expected']}" + (f" +{ra['extra']}" if ra["extra"] else "")
        q_txt = f"{q['matched']}/{q['expected']}" + (f" +{q['extra']}" if q["extra"] else "")
        print(f"{f.stem:<9}{rate_txt:>14}{pct(*r['contract']):>9}{pct(*r['policies']):>10}"
              f"{pct(*r['sources']):>9}{q_txt:>11}{'%d/%d' % r['not_loaded']:>12}")
        for p in r["problems"] + r["notes"]:
            print(f"{'':<9}- {p}")
        tot["rate_ok"] += ra["correct"]; tot["rate_exp"] += ra["expected"]; tot["rate_extra"] += ra["extra"]
        tot["c_ok"] += r["contract"][0]; tot["c_n"] += r["contract"][1]
        tot["p_ok"] += r["policies"][0]; tot["p_n"] += r["policies"][1]
        tot["s_ok"] += r["sources"][0]; tot["s_n"] += r["sources"][1]
        tot["q_ok"] += q["matched"]; tot["q_n"] += q["expected"]; tot["q_extra"] += q["extra"]
        tot["n_ok"] += r["not_loaded"][0]; tot["n_n"] += r["not_loaded"][1]
    print()
    print(f"Rates correct      {pct(tot['rate_ok'], tot['rate_exp'])}  ({tot['rate_ok']}/{tot['rate_exp']}, {tot['rate_extra']} not expected)")
    print(f"Contract details   {pct(tot['c_ok'], tot['c_n'])}")
    print(f"Policies           {pct(tot['p_ok'], tot['p_n'])}")
    print(f"Sources            {pct(tot['s_ok'], tot['s_n'])}  (of correct values)")
    print(f"Questions          {tot['q_ok']}/{tot['q_n']} asked, {tot['q_extra']} not needed")
    print(f"Not loaded         {tot['n_ok']}/{tot['n_n']} found")


if __name__ == "__main__":
    main()
