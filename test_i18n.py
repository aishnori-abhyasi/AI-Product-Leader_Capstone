"""Language-support tests. Run: python test_i18n.py"""
import re
import i18n
from i18n import LANGS, UI, NOTICE, REASON, RECOURSE, KB_I18N, REVIEW, localized_script, nums
from indic import aksharas, readability_indic
from matching_engine import FEATURES
from script_rag import KB
from compliance import notice
from app import create_app, PAGE
from test_app import customer, analyse, hdr

NON_EN = ["hi", "ta", "te"]
INDIC = {"hi": r"[\u0900-\u097F]", "ta": r"[\u0B80-\u0BFF]", "te": r"[\u0C00-\u0C7F]"}

def main():
    ok = lambda n: print("PASS", n)
    # 1 catalog completeness
    for l in NON_EN:
        assert set(UI[l]) == set(UI["en"]), (l, set(UI["en"]) ^ set(UI[l]))
        assert set(i18n.OCC[l]) == set(i18n.OCC["en"]) and set(i18n.LEVELS[l]) == set("ABCD") and set(i18n.KIND_VERB[l]) == set(i18n.KIND_VERB["en"])
        assert set(REASON[l]) == set(FEATURES) and all(len(v) == 2 for v in REASON[l].values())
        assert set(RECOURSE[l]) == {(f, e) for f in ("bill_on_time_ratio", "low_balance_days_ratio") for e in (True, False)}
        assert set(NOTICE[l]) | {"version", "grievance_contact"} == set(notice()), (l, set(notice()) ^ set(NOTICE[l]))
        assert {d["id"] for d in KB} == set(KB_I18N[l]), l
        for key in ("credit", "no_credit"): assert i18n.HEADLINE[l][key]
    ok("every key exists in every language (UI, notice, reasons, recourse, schemes, occupations)")

    # 2 translation drift guard: numbers and URL must be identical to the English facts
    for l in NON_EN:
        for d in KB:
            assert nums(KB_I18N[l][d["id"]][1]) == nums(d["text"]), (l, d["id"], nums(KB_I18N[l][d["id"]][1]), nums(d["text"]))
        assert "register.eshram.gov.in" in KB_I18N[l]["eshram"][1]
        assert nums(NOTICE[l]["how_long"]) == nums(notice()["how_long"]) == {"30"} and nums(NOTICE[l]["period"]) == {"90"}
    caveat = {"hi": "रेहड़ी-पटरी", "ta": "தெருவோர", "te": "వీధి"}
    for l in NON_EN: assert caveat[l] in KB_I18N[l]["svanidhi"][1] and caveat[l] in KB_I18N[l]["svanidhi"][0]
    ok("numbers, amounts, URL and the 'street vendors only' caveat are preserved in every translation")

    # 3 UI wiring: every data-i key and every S.key used by the page exists
    used = set(re.findall(r'data-i="(\w+)"', PAGE)) | set(re.findall(r"\bS\.(\w+)", PAGE))
    assert used <= set(UI["en"]), used - set(UI["en"]); ok(f"page uses {len(used)} catalog keys, all defined")

    # 4 akshara counting
    assert aksharas("कमाई", "hi") == 3 and aksharas("प्रधानमंत्री", "hi") == 5 and aksharas("తెలుగు", "te") == 3 and aksharas("வணக்கம்", "ta") == 5
    ok("akshara counter on known words (Hindi conjuncts, Telugu, Tamil)")

    # 5 localized script + review gate
    kb_ver = [{**d, "verified": True} for d in KB]
    match = {"offers": [{"id": "svanidhi"}, {"id": "apy"}, {"id": "pmsby"}]}
    for l in NON_EN:
        r = localized_script(match, l, kb_ver)
        assert r["grounded"] and r["meaning_ok"] and re.search(INDIC[l], r["script"]) and r["grade"] is None
        assert r["safe_to_ship"] is False and r["translation_reviewed"] is False        # not reviewed -> never shippable
        REVIEW[l] = "Native Reviewer"; assert localized_script(match, l, kb_ver)["safe_to_ship"] is True
        assert localized_script(match, l, KB)["safe_to_ship"] is False                  # unverified facts still block
        REVIEW[l] = None
    gap = localized_script({"offers": [{"id": "svanidhi"}]}, "hi", [{**KB[2], "id": "svanidhi"}])
    saved = KB_I18N["hi"].pop("svanidhi"); r = localized_script({"offers": [{"id": "svanidhi"}]}, "hi", kb_ver); KB_I18N["hi"]["svanidhi"] = saved
    assert r["meaning_ok"] is False and r["safe_to_ship"] is False
    ok("localized scripts: grounded; blocked until a native reviewer signs off AND facts are verified; missing translation blocks")

    # 6 advisory readability report (uncalibrated)
    for l in NON_EN:
        r = localized_script({"offers": [{"id": "eshram"}, {"id": "svanidhi"}, {"id": "apy"}]}, l, kb_ver)["readability"]
        print(f"      {l}: {r['avg_sentence_words']} words/sentence, {r['avg_aksharas_per_word']} aksharas/word, long words {r['long_word_share']:.0%}, "
              f"within advisory targets: {r['within_targets']}, sentences to review: {len(r['long_sentences'])}")
    ok("Indic readability computed (advisory only - uncalibrated, needs comprehension testing)")

    # 7 end-to-end through the API
    app = create_app(env="sandbox", n_workers=400); c = app.test_client()
    assert c.get("/api/i18n/xx").status_code == 404 and c.get("/api/i18n/ta").get_json()["ui"]["send"] == UI["ta"]["send"]
    assert c.get("/api/notice?lang=te").get_json()["why"] == NOTICE["te"]["why"] and c.get("/api/notice").get_json()["why"] == notice()["why"]
    for i, l in enumerate(NON_EN):
        tok = customer(c, f"90000001{i}0")
        r = analyse(c, tok, "mixed.arun@upi", occ="street vendor", lang=l); assert r.status_code == 200, r.get_json()
        j = r.get_json()
        assert j["lang"] == l and j["level"] in i18n.LEVELS[l].values() and j["disclosure"] == i18n.DISCLOSURE[l]
        assert j["headline"] in i18n.HEADLINE[l].values() and re.search(INDIC[l], j["script"]) and re.search(INDIC[l], j["next_steps"][0]["name"] + j["next_steps"][0]["action"])
        allowed = {x[0 if h else 1] for x in REASON[l].values() for h in (True, False)} if False else {t for v in REASON[l].values() for t in v}
        assert all(w["text"] in allowed for w in j["why"]), j["why"]
        assert re.search(caveat[l], j["script"]) or "svanidhi" not in str(j["next_steps"]).lower()
        cons = c.get("/api/my/data", headers=hdr(tok)).get_json()["consents"][0]; assert cons["language"] == l
    en = analyse(c, customer(c, "9000000199"), "mixed.arun@upi", occ="street vendor", lang="xx").get_json(); assert en["lang"] == "en"
    ok("hi/ta/te end-to-end: localized level, headline, steps, reasons, script, disclosure; language recorded on consent; unknown -> English")
    print("\nALL I18N TESTS PASSED")

if __name__ == "__main__": main()
