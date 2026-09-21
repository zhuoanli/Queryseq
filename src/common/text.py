"""Finding queries and their embeddings.

QuerySeq routes on a *text* description of the finding being asked about.  That
only counts as a contribution if the routing follows the semantics of the query
rather than its identity, so this module deliberately provides several
meaning-preserving phrasings per finding (`PARAPHRASE_FAMILIES`) plus the
machinery to build meaning-*breaking* controls (shuffled and counterfactual
queries) in query_diagnostics.py.

Encoder is BiomedVLP-CXR-BERT-specialized loaded as `BertModel`.  Loading it
through `AutoModel` with trust_remote_code returns the CXR-BERT wrapper, whose
weights nest under `.bert`; a state dict then loads nothing and you get a
randomly initialised text tower that fails silently.  The corpus pipeline hit this.
"""
import json
import os

import numpy as np

from .data import CACHE, PB, load_labels

TEXT_ENCODER = "microsoft/BiomedVLP-CXR-BERT-specialized"
TEXT_DIM = 768

# Short clinical term per finding.  The first 32 come from the curated ontology
# shipped with the corpus (data/text/pathologies.json); the remainder are the
# radiological terms for label columns that file does not cover.
_EXTRA_TERMS = {
    "Cerebral infarction": "cerebral infarct",
    "Ventriculomegaly": "ventriculomegaly",
    "Mastoiditis": "mastoiditis",
    "Cerebral hemorrhage": "intracerebral hemorrhage",
    "Lipoma of brain": "intracranial lipoma",
}

# One-clause descriptions, used as the `definition` paraphrase.  Written for the
# findings that are evaluable at the prevalence floor; anything not listed falls
# back to a generic phrasing rather than an invented definition.
_DEFINITIONS = {
    "Gliosis": "glial scarring of brain parenchyma",
    "Cerebral atrophy": "loss of brain parenchymal volume",
    "Ventriculomegaly": "enlargement of the cerebral ventricles",
    "Cerebral edema": "increased water content in brain tissue",
    "Cerebral infarction": "ischemic tissue death in the brain",
    "Empty sella syndrome": "flattened pituitary with cerebrospinal fluid in the sella",
    "Arachnoid cyst": "cerebrospinal fluid collection within the arachnoid membrane",
    "Metastatic malignant neoplasm to brain": "secondary tumour deposits in the brain",
    "Demyelinating disease of central nervous system": "loss of myelin in white matter",
    "Mastoiditis": "inflammatory opacification of the mastoid air cells",
    "Cerebellar degeneration": "loss of cerebellar volume",
    "Cerebral hemorrhage": "bleeding within the brain parenchyma",
    "Encephalomalacia": "softening and cavitation of brain tissue after injury",
    "Lacunar infarct": "small deep infarct from small vessel occlusion",
    "Intracranial meningioma": "extra-axial tumour arising from the meninges",
    "Chronic mastoiditis": "long-standing mastoid inflammation",
    "Silent micro-hemorrhage of brain": "small foci of chronic blood products",
    "Cyst of pineal gland": "fluid-filled cyst in the pineal region",
    "Structure of cave of septum pellucidum": "fluid space between the leaves of the septum pellucidum",
    "Mega cisterna magna": "enlarged posterior fossa cerebrospinal fluid space",
    "Herniation of nucleus pulposus": "displacement of intervertebral disc material",
    "Cavernous hemangioma": "vascular malformation with blood products",
    "Spinal cord compression": "narrowing that indents the spinal cord",
    "Hyperostosis of skull": "thickening of the calvarial bone",
    "Glioma": "primary tumour of glial origin",
    "Chiari malformation": "caudal displacement of the cerebellar tonsils",
    "Intracranial aneurysm": "focal outpouching of an intracranial artery",
    "Subdural intracranial hemorrhage": "blood collection beneath the dura",
    "Pituitary adenoma": "benign tumour of the pituitary gland",
    "Choroid plexus cyst": "cyst within the choroid plexus",
    "Schwannoma": "nerve sheath tumour",
    "Watershed infarct": "infarct at the border zone between arterial territories",
}

# Meaning-preserving rewordings.  Routing should be stable across these; if it
# is not, the model is keying on surface form and the language claim is dropped.
PARAPHRASE_FAMILIES = ["name", "term", "sentence", "question", "definition"]


def _terms():
    raw = json.load(open(f"{PB}/data/text/pathologies.json"))
    raw = raw.get("pathologies", raw)
    out = {}
    for k, v in raw.items():
        # 'There is cerebral atrophy' -> 'cerebral atrophy'
        out[k] = v["positive"].replace("There is ", "").strip()
    out.update(_EXTRA_TERMS)
    return out


def query_variants(findings=None):
    """{finding: {family: query string}} for every label column."""
    if findings is None:
        findings = list(load_labels().columns)
    terms = _terms()
    out = {}
    for f in findings:
        t = terms.get(f, f.lower())
        d = _DEFINITIONS.get(f)
        out[f] = {
            "name": f,
            "term": t,
            "sentence": f"There is {t}.",
            "question": f"Is there evidence of {t}?",
            "definition": d if d else f"imaging finding of {t}",
        }
    return out


def _encode(strings, device=None, batch=64):
    import torch
    from transformers import AutoTokenizer, BertModel
    if os.environ.get("HF_HOME"):
        os.environ.setdefault(
            "HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TEXT_ENCODER, trust_remote_code=True)
    mdl = BertModel.from_pretrained(TEXT_ENCODER, add_pooling_layer=False)
    mdl.eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(strings), batch):
            t = tok(strings[i:i + batch], padding=True, truncation=True,
                    max_length=64, return_tensors="pt")
            t = {k: v.to(device) for k, v in t.items()}
            h = mdl(**t).last_hidden_state[:, 0]
            h = torch.nn.functional.normalize(h, dim=-1)
            out.append(h.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def finding_text_emb(family="sentence", findings=None, rebuild=False):
    """(F, 768) L2-normalised query embeddings, cached to disk.

    Cheap enough to recompute (a few hundred short strings) but cached so that
    CPU-only analysis jobs never need to load a BERT.
    """
    if findings is None:
        findings = list(load_labels().columns)
    path = f"{CACHE}/qemb_{family}.npy"
    keys = f"{CACHE}/qemb_{family}_keys.json"
    if os.path.exists(path) and not rebuild:
        cached = json.load(open(keys))
        if cached == list(findings):
            return np.load(path)
    qv = query_variants(findings)
    strings = [qv[f][family] for f in findings]
    E = _encode(strings)
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, E)
    with open(keys, "w") as fh:
        json.dump(list(findings), fh)
    return E


def build_all(rebuild=False):
    return {fam: finding_text_emb(fam, rebuild=rebuild)
            for fam in PARAPHRASE_FAMILIES}
