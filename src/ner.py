
import re
import json
from transformers import pipeline
from config import USE_NER
from prepare_data import load_dataset
from nltk.corpus import stopwords
arabic_stopwords = stopwords.words('arabic')


def load_ner_model():

    ner = pipeline(
        "token-classification",
        model="hatmimoha/arabic-ner",
        aggregation_strategy="simple",
        device=0
    )

    return ner



def truncate_to_512_tokens(text, tokenizer, max_tokens=512):
    """
    Truncate text to ner model token limit safely.
    """

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=True
    )

    return tokenizer.decode(
        encoded["input_ids"],
        skip_special_tokens=True
    )






def remove_sub_entities(entities):
    """
    Keep only the most informative entities.
    Example:
        ['فاضل', 'فاضل نوفاليتش']
    becomes:
        ['فاضل نوفاليتش']
    """

    entities = sorted(set(entities), key=len, reverse=True)

    result = []

    for entity in entities:
        if not any(
            entity != other and entity in other
            for other in result
        ):
            result.append(entity)

    return sorted(result)



def clean_entities(entity_list):
    """
    Remove common Arabic ner noise.
    """

    cleaned = []

    for entity in entity_list:

        entity = entity.strip()

        # remove wordpiece fragments
        if "##" in entity:
            continue

        # remove tiny fragments
        if len(entity) < 4:
            continue

        # remove latin-only garbage
        if re.fullmatch(r"[A-Za-z]+", entity):
            continue

        # normalize spaces
        entity = " ".join(entity.split())

        cleaned.append(entity)

    cleaned = remove_sub_entities(cleaned)

    return sorted(set(cleaned))




def extract_entities_from_ner(
    ner_output
):
    """
    Clean Arabic ner extraction for RAG.
    Works with aggregation_strategy='simple'
    """

    entities = {
        "PERSON": [],
        "ORGANIZATION": [],
        "LOCATION": [],
        "DATE": []
    }

    label_map = {
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "DATE": "DATE",
        "PERSON": "PERSON",
        "ORGANIZATION": "ORGANIZATION",
        "LOCATION": "LOCATION"
    }

    for item in ner_output:


        word = item.get("word", "").strip()
        label = item.get("entity_group", "").strip()

        if not word or not label:
            continue

        # normalize label
        label = label.replace("B-", "").replace("I-", "")

        if label not in label_map:
            continue

        entity_type = label_map[label]

        # CLEAN ENTITY TEXT
        word = word.replace("##", "")
        word = " ".join(word.split())

        # filter garbage
        if len(word) < 3:
            continue

        entities[entity_type].append(word)

    return {
        "PERSON": clean_entities(entities["PERSON"]),
        "ORGANIZATION": clean_entities(entities["ORGANIZATION"]),
        "LOCATION": clean_entities(entities["LOCATION"]),
        "DATE": sorted(set(entities["DATE"]))
    }






def build_ner( ner_output,text):
    """
    Builds ner from structured ner + text.
    """

    persons = ner_output.get("PERSON", [])
    organizations = ner_output.get("ORGANIZATION", [])
    locations = ner_output.get("LOCATION", [])
    dates = ner_output.get("DATE", [])

#    keywords = extract_keywords(text,disambiguator=mle)

    ner = {

        "persons": persons,
        "organizations": organizations,
        "locations": locations,
        "dates": dates,
    }

    return ner


 
import json


def generate_ner(
    text,
    ner_model,
    max_tokens=512,
):
    """
    Generate NER metadata from a single text.

    Parameters
    ----------
    text : str
        Input text.

    ner_model :
        Loaded NER pipeline/model.

    max_tokens : int
        Maximum number of tokens for NER model.

    doc_id : int
        Optional document ID.

    Returns
    -------
    dict
        NER metadata dictionary.
    """

    if not text or not text.strip():
        return {}

    try:
        tokenizer = ner_model.tokenizer

        safe_text = truncate_to_512_tokens(
            text,
            tokenizer,
            max_tokens=max_tokens
        )

        ner_output_raw = ner_model(safe_text)

        ner_output = extract_entities_from_ner(
            ner_output_raw
        )

        ner_metadata = build_ner(
            ner_output=ner_output,
            text=text
        )

        return ner_metadata

    except Exception as e:
        print(f"NER error: {e}")
        return {}




def add_ner_to_text(
    text,
    ner=None,
    use_ner=USE_NER,
):
    """
    Merge NER metadata with text for embedding.

    Parameters
    ----------
    text : str
        Original text.

    ner : dict
        NER metadata.

    use_ner : bool
        If False, return original text only.

    Returns
    -------
    str
        Text prepared for embedding.
    """

    if not use_ner or not ner:
        return text

    field_names = {
        "persons": "الأشخاص",
        "organizations": "المنظمات",
        "locations": "الأماكن",
        "dates": "التواريخ",
        "keywords": "الكلمات المفتاحية",
    }

    sections = []

    for key, title in field_names.items():

        value = ner.get(key)

        if value is None:
            continue

        # Decode JSON strings
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                pass

        # Skip empty values
        if value in ("", [], {}, ()):
            continue

        # List handling
        if isinstance(value, list):

            value = "، ".join(
                map(str, value)
            )

        # Dictionary handling
        elif isinstance(value, dict):

            value = "، ".join(
                f"{k}: {', '.join(map(str, v)) if isinstance(v, list) else v}"
                for k, v in value.items()
            )

        sections.append(
            f"{title}: {value}"
        )

    sections.append(
        f"النص: {text}"
    )

    return "\n".join(sections)




if __name__ == "__main__":

    ner_model = load_ner_model()

    text = """
    احمد ابن حنبل ولد عام 1440م في قبيلة البدون.
    ولاية سطيف، هي ولاية جزائرية تقع في شمال شرق الجزائر،
    تحمل عاصمتها نفس الاسم سطيف.
    """

    ner_metadata = generate_ner(
        text=text,
        ner_model=ner_model,
        max_tokens=512
    )

    text = add_ner_to_text(
        text=text,
        ner=ner_metadata,
        use_ner=USE_NER
    )

    print("NER:")
    print(ner_metadata)

    print(" text:")
    print(text)
