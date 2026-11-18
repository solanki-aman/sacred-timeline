"""The standard tag set asked of every moment when a clip is indexed.

These are fixed so that every indexed moment can be filtered the same way (`--where emotion=angry`).
Free-form searches are separate: they are asked on demand and cached (see search.py).
"""

EMOTIONS = {
    "happy": "Happy or delighted",
    "amused": "Amused, finding something funny",
    "sad": "Sad or hurt",
    "angry": "Angry or irritated",
    "anxious": "Afraid, anxious or worried",
    "surprised": "Surprised or shocked",
    "awkward": "Awkward or embarrassed",
    "confident": "Confident or self-assured",
}
VOICES = {
    "calm": "Calm and even",
    "serious": "Serious or matter-of-fact",
    "warm": "Warm or friendly",
    "nervous": "Nervous or hesitant",
    "sarcastic": "Sarcastic or mocking",
    "deadpan": "Deadpan or flat",
    "excited": "Excited or eager",
    "pleading": "Pleading or desperate",
    "angry": "Angry or raised",
}

TAGS = {
    "footage": {"type": "noul", "instructions":
                "Is this live-action footage from the film or show, rather than a promotional end card, "
                "subscribe screen or channel graphic?"},
    "people": {"type": "choice", "instructions": "How many people can be seen?",
               "criteria": {"none": "No people", "one": "One person", "two": "Two people",
                            "group": "Three or more people"}},
    "shot": {"type": "choice", "instructions": "How is the shot framed?",
             "criteria": {"close": "Close-up on a face", "medium": "Medium shot, people from the waist up",
                          "wide": "Wide shot showing the whole place"}},
    "setting": {"type": "choice", "instructions": "Where does this take place?",
                "criteria": {"interior": "Indoors", "exterior": "Outdoors", "vehicle": "Inside a vehicle"}},
    "speaking": {"type": "noul", "instructions": "Is someone speaking in this moment?"},
    "emotion": {"type": "choice", "instructions":
                "Which emotion does the main character in this moment show most, from their face, body and voice? "
                "Pick the closest even if it is subtle.", "criteria": EMOTIONS},
    "intensity": {"type": "score", "instructions": "How strongly is that emotion shown?",
                  "criteria": ["Subtle", "Clear", "Strong"]},
    "voice": {"type": "choice", "instructions": "What is the tone of the voice that is speaking?",
              "criteria": VOICES},
    "tension": {"type": "noul", "instructions":
                "Is there tension, conflict or discomfort between the characters in this moment?"},
    "funny": {"type": "noul", "instructions": "Is this moment meant to be funny?"},
    "action": {"type": "noul", "instructions": "Is there physical action or fast movement, like a fight, chase or crash?"},
}


def flatten(answers: dict[str, dict]) -> dict[str, float | str]:
    """One scalar per tag, for filtering: noul -> probability, choice -> key, score -> value."""
    out = {}
    for k, a in answers.items():
        if "noul" in a:
            out[k] = round(a["noul"], 3)
        elif "choice" in a:
            out[k] = a["choice"]
            out[f"{k}_conf"] = round(a["confidence"], 3)
        elif "score" in a:
            out[k] = round(a["score"], 2)
    return out
