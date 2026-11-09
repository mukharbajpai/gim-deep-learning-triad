import torch
from src.vae_fashion import ConvAE, ConvVAE, loss_terms
from src.agentic_qa import safe_calculate, normalize, exact_match, token_f1
from src.transformer_sentiment import prediction_metrics


def test_vae_shapes_and_finite_loss():
    x = torch.rand(2, 1, 28, 28)
    model = ConvVAE(8)
    rec, mu, lv = model(x)
    assert rec.shape == x.shape and mu.shape == (2, 8) and lv.shape == (2, 8)
    r, kl = loss_terms(rec, x, mu, lv)
    assert r.isfinite() and kl.isfinite() and kl >= 0
    baseline_rec = ConvAE(8)(x)
    assert baseline_rec.shape == x.shape and baseline_rec.isfinite().all()
    model.eval()
    assert torch.equal(model(x)[0], model(x)[0])


def test_calculator_safe():
    assert safe_calculate('2 * (3 + 4)') == 14
    for bad in ('__import__("os")', '2**10', '[1,2]'):
        try: safe_calculate(bad)
        except (ValueError, SyntaxError): pass
        else: raise AssertionError(f'Allowed unsafe syntax: {bad}')


def test_qa_metrics():
    assert normalize('The Eiffel Tower!') == 'eiffel tower'
    assert exact_match('The Eiffel Tower!', ['eiffel tower']) == 1
    assert token_f1('Paris France', ['Paris']) > 0


def test_classification_metrics_cover_imbalance():
    scores = prediction_metrics([0, 0, 1, 1], [0, 0, 0, 0])
    assert scores['accuracy'] == 0.5
    assert scores['binary_f1'] == 0.0
    assert 0 < scores['macro_f1'] < scores['accuracy']
