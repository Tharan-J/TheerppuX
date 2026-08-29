"""
Tests for Evaluation Metrics, Entity Preservation, Terminology, and Error Analysis.
"""

from src.evaluation.bleu import BLEUMetric
from src.evaluation.chrf import ChrFMetric
from src.evaluation.rouge import ROUGEMetric
from src.evaluation.entity_metrics import EntityPreservationMetric
from src.evaluation.terminology import LegalTerminologyMetric
from src.analysis.error_analysis import LegalErrorClassifier, Severity, ErrorTaxonomy
from src.analysis.consistency import CrossLanguageConsistencyChecker


def test_bleu_and_chrf_metrics():
    bleu = BLEUMetric()
    chrf = ChrFMetric()

    preds = ["சென்னை உயர் நீதிமன்றம் தீர்ப்பு வழங்கியது."]
    refs = [["சென்னை உயர் நீதிமன்றம் தீர்ப்பு வழங்கியது."]]

    bleu_res = bleu.compute(preds, refs)
    assert bleu_res["bleu"] == 100.0

    chrf_res = chrf.compute(preds, refs)
    assert chrf_res["chrf_plus_plus"] > 95.0


def test_rouge_metric():
    rouge = ROUGEMetric()
    preds = ["The court allowed the appeal."]
    refs = ["The court allowed the appeal."]

    res = rouge.compute(preds, refs)
    assert res["rouge_l"] == 100.0
    assert "Supplementary" in res["note"]


def test_number_preservation_and_corruption():
    entity_metric = EntityPreservationMetric()

    src = ["The accused was ordered to pay a fine of ₹50,000 on 12-03-2024 under Section 302."]
    # Perfect candidate
    cand_perfect = ["எதிரிக்கு பிரிவு 302-ன் கீழ் 12-03-2024 அன்று ₹50,000 அபராதம் விதிக்கப்பட்டது."]
    res_perfect = entity_metric.compute(src, cand_perfect, target_lang="ta")
    assert res_perfect["number_accuracy"] == 100.0

    # Corrupted candidate (₹50,000 -> ₹5,000)
    cand_corrupted = ["எதிரிக்கு பிரிவு 302-ன் கீழ் ₹5,000 அபராதம் விதிக்கப்பட்டது."]
    res_corrupted = entity_metric.compute(src, cand_corrupted, target_lang="ta")
    assert res_corrupted["number_accuracy"] < 100.0
    assert res_corrupted["corrupted_number_count"] > 0


def test_legal_terminology_and_mistranslation():
    terms_dict = {
        "conviction": {
            "canonical": "தண்டனை",
            "variants": ["தண்டனை", "தண்டிக்கப்பட்டார்"],
        },
        "petitioner": {
            "canonical": "மனுதாரர்",
            "variants": ["மனுதாரர்", "மனுதாரரின்"],
        },
    }
    metric = LegalTerminologyMetric(legal_terms_dict=terms_dict)

    src = ["The conviction of the petitioner is upheld."]
    cand_correct = ["மனுதாரரின் தண்டனை உறுதி செய்யப்படுகிறது."]
    res = metric.compute(src, cand_correct, target_lang="ta")

    assert res["legal_term_accuracy"] == 100.0
    assert res["correct_terms"] == 2


def test_error_classifier_critical_errors():
    classifier = LegalErrorClassifier()

    # Test meaning reversal (convicted -> acquitted)
    src = ["The appellant was convicted under Section 302."]
    cand = ["மேல்முறையீட்டாளர் விடுதலை செய்யப்பட்டார்."]  # acquitted

    res = classifier.analyze(src, cand, target_lang="ta")
    assert res["critical_errors"] > 0
    reversal_errors = [e for e in res["classified_errors"] if e["error_type"] == ErrorTaxonomy.E10_MEANING_REVERSAL.value]
    assert len(reversal_errors) > 0
    assert reversal_errors[0]["severity"] == Severity.CRITICAL.value


def test_cross_language_consistency():
    checker = CrossLanguageConsistencyChecker()

    src = ["The fine of ₹50,000 was imposed under Section 302."]
    ta = ["பிரிவு 302-ன் கீழ் ₹50,000 அபராதம் விதிக்கப்பட்டது."]
    ml = ["വകുപ്പ് 302 പ്രകാരം ₹50,000 പിഴ ചുമത്തി."]

    res = checker.compare_translations(src, ta, ml)
    assert res["cross_language_consistency_score"] == 100.0
    assert res["discrepancy_count"] == 0
