import kagglehub
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import evaluate
import torch
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
import spacy
import random
import re

# Load spaCy English model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Contraction normalization map
contraction_map = {
    "can't": "can not",
    "won't": "will not",
    "don't": "do not",
    "didn't": "did not",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    # add more as needed
}

def normalize_contractions(text):
    for contraction, full in contraction_map.items():
        # Use word boundaries and case-insensitive replacement
        pattern = re.compile(r'\b' + re.escape(contraction) + r'\b', flags=re.IGNORECASE)
        text = pattern.sub(full, text)
    return text

phrase_mapping = {
    "thank you": ["I thank thee", "I thank ye", "Gramercy"],
    "good morning": ["Good morrow", "Morrow to thee"],
    "good evening": ["Good e'en", "Even so"],
    "do you": ["dost thou"],
    "will you": ["wilt thou"],
    "you are": ["thou art"],
    "do not": ["dost not"],
    "did not": ["didst not"],
    "that is": ["that is", "that be"],
    "what is": ["what is", "what be"],
    "who is": ["who is", "who be"],

    "oh no": ["Alack", "Alas"],
    "oh dear": ["Alack-a-day"],
    "oh my": ["Zounds", "Marry"],

    "it is": ["'tis"],
    "it was": ["'twas"],
    "it were": ["'twere"],
    "it would": ["'twould"],
    "it shall": ["'tshall"],
    "could not": ["couldst not"],
    "would not": ["wouldst not"],
    "should not": ["shouldst not"],
    "can not": ["canst not"],
    "may not": ["mayst not"],
    "shall not": ["shalt not"],
    "will not": ["wilt not"],

    "i am": ["I am", "I be"],
    "you are": ["thou art"],
    "he is": ["he is", "he be"],
    "she is": ["she is", "she be"],
    "they are": ["they are", "they be"],
    "we are": ["we are", "we be"],
    "there is": ["there is", "there be"],
    "there are": ["there are", "there be"],
    "here is": ["here is", "here be"],
    "here are": ["here are", "here be"],

    "my friend": ["mine own friend", "my good friend"],
    "my lord": ["mine own lord", "my good lord"],
    "my lady": ["mine own lady", "my good lady"],
    "my love": ["mine own love", "my true love"],
    "excuse me": ["I crave thy pardon", "Prithee pardon me"],
    "are you": ["art thou"],
    "can you": ["canst thou"],
    "should you": ["shouldst thou"],
    "how are you": ["How fares thee?", "How dost thou fare?"],
    "good day": ["Good morrow", "Good den"],
    "what do you mean": ["What meanst thou?"],
    "i don't know": ["I wot not"],
    "i think so": ["Methinks so"],
    "it seems": ["It doth seem", "It seemeth"],
    "never mind": ["No matter"],
    "of course": ["Ay, marry", "Verily"],
    "go on": ["Proceed", "On with thee"],
    "come here": ["Come hither", "Come hitherward"],
    "stand aside": ["Stand aside", "Stand thee aside"],
    "be quiet": ["Hold thy peace", "Hush"],
    "by god": ["By my troth", "By my faith"],
    "what's wrong": ["What troubles thee?", "What aileth thee?"],
    "don't worry": ["Fret not", "Cease thy fretting"],
    "i assure you": ["I warrant thee", "I assure thee"],
    "rest assured": ["Be thou assured"],

    "what's up": ["What aileth thee?", "What news?"],
    "go away": ["Away with thee"],
    
}

def phrase_replace(text, mapping):
    phrases = sorted(mapping.keys(), key=len, reverse=True)
    for phrase in phrases:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        def replace_func(match):
            repl = mapping[phrase]
            if isinstance(repl, list):
                repl = random.choice(repl)
            # Preserve capitalization style
            if match.group(0).istitle():
                repl = repl.capitalize()
            elif match.group(0).isupper():
                repl = repl.upper()
            return repl
        text = pattern.sub(replace_func, text)
    return text


# Map dictionary for single words
modern_to_shakespeare = {
    "you": 'thou',
    "your": "thy",
    "yours": "thine",
    "yourself": "thyself",
    "i": "I",
    "do": "dost",
    "does": "doth",
    "have": "hast",
    "has": "hath",
    "are": "art",
    "were": "wert",
    
    # ... extended from contractions
    "never": "ne'er",
    "ever": "e'er",
    "over": "o'er",
    "before": "ere",
    "often": "oft",
    "perhaps": "perchance",
    "maybe": "haply",
    "soon": "anon",

    # ... extended from emotions
    "oh": "O",
    "wow": "Marry",
    
    # Miscellaneous
    "stop": "cease",
    "stop": "hold",
    "listen": "hark",
    "listen": "mark me",
    "help me": "aid me",
    "indeed": "forsooth",
    "yes": "yea",
    "no": "nay",
    "think": "bethink",
    "know": "wot",
    "understand": "ken",
    "give": "bestow",
    "go": "hie",
    "come": "approach",
    "leave": "depart",
    "run": "hasten",
    "say": "speak",
    "says": "saith",
    "said": "quoth",
    "ask": "beseech",
    "tell": "relate",
    "eat": "feast",
    "drink": "quaff",
    "sleep": "slumber",
    "fight": "duel",
    "work": "toil",
    "man": "gentleman",
    "woman": "gentlewoman",
    "boy": "lad",
    "girl": "lass",
    "child": "bairn",
    "money": "coin",
    "food": "victuals",
    "house": "dwelling",
    "home": "home",
    "town": "burgh",
    "city": "city",
    "country": "realm",
    "world": "world",
    "hate": "hate",
    "joy": "mirth",
    "sadness": "sorrow",
    "anger": "wrath",
    "fear": "dread",
    "brave": "valiant",
    "cowardly": "craven",
    "wise": "sage",
    "foolish": "foolish",
    "beautiful": "fair",
    "ugly": "ill-favored",
    "good": "good",
    "bad": "ill",
    "truth": "sooth",
    "lie": "falsehood",
    "friend": "comrade",
    "enemy": "foe",
    "knight": "knight",
    "doctor": "physician",
    "teacher": "tutor",
    "student": "scholar",
    "book": "tome",
    "letter": "missive",
    "sword": "blade",
    "shield": "buckler",
    "horse": "steed",
    "road": "path",
    "forest": "wood",
    "river": "stream",
    "mountain": "mount",
    "sea": "ocean",
    "sky": "heavens",
    "star": "star",
    "day": "day",
    "night": "night",
    "morning": "morn",
    "evening": "eve",
    "today": "this day",
    "tomorrow": "morrow",
    "yesterday": "yestreen",
    "now": "anon",
    "then": "then",
    "always": "ever",
    "never": "ne'er",
    "often": "oft",
    "seldom": "seldom",
    "here": "hither",
    "there": "thither",
    "where": "whither",
    "something": "aught",
    "nothing": "naught",
    "big": "great",
    "small": "little",
    "fast": "swift",
    "strong": "sturdy",
    "weak": "feeble",
    "dirty": "foul",
    "difficult": "arduous",
    "finished": "done",
    "start": "begin",
    "end": "finish",
    "goodbye": "adieu",
    "joy": "mirth",
    "sorrow": "sorrow",
    "happiness": "bliss",
    "misery": "woe",
    "courage": "valor",
    "fear": "dread",
    "ghost": "spectre",
    "dream": "dream",
    "reality": "truth",
    "fantasy": "fancy",
    "story": "tale",
    "poem": "verse",
    "song": "lay",
    "music": "melody",
    "science": "knowledge",
    "punishment": "chastisement",
    "reward": "recompense",
    "king": "Monarch",
    "queen": "Consort",
    "prince": "Heir",
    "princess": "Maiden",
    "lord": "Noble",
    "lady": "Dame",
    "servant": "minion",
    "master": "Master",
    "mistress": "Mistress",
    "friendship": "amity",
    "enmity": "malice",
    "love": "have affection for",
    "hatred": "spite",
    "peace": "concord",
    "war": "strife",
    "battle": "conflict",
    "victory": "triumph",
    "defeat": "rout",
    "courage": "metle",
    "fear": "timidity",
    "strength": "might",
    "weakness": "infirmity",
    "beauty": "comeliness",
    "ugliness": "deformity",
    "wealth": "riches",
    "poverty": "want",
    "health": "vigor",
    "sickness": "ailment",
    "life": "existence",
    "death": "demise",
    "world": "cosmos",
    "earth": "terra",
    "heaven": "celestial sphere",
    "hell": "underworld",
    "sun": "sol",
    "moon": "luna",
    "star": "luminary",
    "day": "daylight",
    "night": "nightfall",
    "morning": "dawn",
    "evening": "dusk",
    "yesterday": "yester-morn",
    "tomorrow": "morrow-tide",
    "now": "presently",
    "then": "at that moment",
    "always": "evermore",
    "never": "nevermore",
    "often": "full oft",
    "seldom": "rarely",
    "here": "in this place",
    "there": "in that place",
    "where": "in what place",
    "everywhere": "in every place",
    "nowhere": "in no place",
    "something": "a certain thing",
    "nothing": "not a thing",
    "anything": "any manner of thing",
    "everything": "all things",
    "someone": "a certain person",
    "no one": "no person",
    "anyone": "any person",
    "everyone": "all persons",
    "another": "an additional",
    "other": "remaining",
    "same": "identical",
    "different": "distinct",
    "new": "novel",
    "old": "ancient",
    "young": "youthful",
    "big": "large",
    "small": "petite",
    "long": "lengthy",
    "short": "brief",
    "wide": "broad",
    "narrow": "confined",
    "deep": "profound",
    "shallow": "superficial",
    "hot": "sultry",
    "cold": "frigid",
    "warm": "tepid",
    "cool": "chilly",
    "wet": "damp",
    "dry": "arid",
    "light": "luminous",
    "dark": "obscure",
    "bright": "radiant",
    "dim": "faint",
    "loud": "resonant",
    "quiet": "silent",
    "fast": "rapid",
    "slow": "leisurely",
    "strong": "robust",
    "glad" : "gleeful",
    "weak": "frail",
    "heavy": "ponderous",
    "light": "feathery",
    "clean": "pure",
    "dirty": "sullied",
    "easy": "effortless",
    "help": "succour",
    "difficult": "arduous",
    "ready": "prepared",
    "finished": "completed",
    "start": "commence",
    "end": "terminate",
    "true": "veritable",
    "false": "untrue",
    "right": "correct",
    "wrong": "incorrect",
    "happy": "joyful",
    "sad": "sorrowful",
    "angry": "wroth",
    "afraid": "Fearful",
    "brave": "courageous",
    "cowardly": "dastardly",
    "wise": "sagacious",
    "foolish": "senseless",
    "beautiful": "beauteous",
    "ugly": "hideous",
    "good": "virtuous",
    "bad": "wicked",
    "sick": "ailing",
    "healthy": "robust",
    "dead": "deceased",
    "alive": "living",
    "rich": "wealthy",
    "poor": "impoverished",
    "hungry": "famished",
    "thirsty": "parched",
    "tired": "weary",
    "rested": "refreshed",
    "sleepy": "drowsy",
    "awake": "awakened",
    "happy": "merrily",
    "sadly": "sadly",
    "angrily": "wrathfully",
    "bravely": "courageously",
    "cowardly": "dastardly",
    "wisely": "sagaciously",
    "foolishly": "senselessly",
    "beautifully": "beauteously",
    "uglily": "hideously",
    "badly": "ill",
    "sickly": "ailingly",
    "healthily": "robustly",
    "deadly": "fatally",
    "lively": "vibrantly",
    "richly": "opulently",
    "poorly": "meagerly",
    "hungrily": "famishedly",
    "thirstily": "parchedly",
    "tiredly": "wearily",
    "restedly": "refreshingly",
    "sleepily": "drowsily",
    "awakely": "awakenedly",
    "hereabouts": "hereabouts",
    "quickly": "swiftly",
    "slowly": "leisurely",
    "strongly": "sturdily",
    "weakly": "feebly",
    "heavily": "ponderously",
    "lightly": "featherily",
    "cleanly": "purely",
    "dirtily": "foully",
    "easily": "effortlessly",
    "difficultly": "arduously",
    "truly": "verily",
    "today": "this day",
    "tomorrow": "the morrow",
    "yesterday": "the yesternight",
    "now": "presently",
    "then": "thence",
    "always": "everlastingly",
    "never": "nevermore",
    "often": "oft",
    "here": "hither",
    "there": "thither",
    "where": "whither",
    
}

# Regex for cleaning space before punctuation
re_space_before_punct = re.compile(r'\s+([,.!?;:])')

def clean_text_spacing(text):
    return re_space_before_punct.sub(r'\1', text)

def pos_aware_substitution(token):
    original_text = token.text
    text = original_text.lower()
    pos = token.pos_

    if pos in ("NOUN", "VERB", "PRON", "ADJ", "ADV", "DET"):
        subs = modern_to_shakespeare.get(text)
        if subs:
            if isinstance(subs, list):
                subs = random.choice(subs)

            # 🔠 Match the original token's case
            if original_text.istitle():
                subs = subs.capitalize()
            elif original_text.isupper():
                subs = subs.upper()

            return subs

    return original_text


# Context-aware Shakespearean starter phrases dict
starters_map = {
    "thank": "I thank thee",
    "hello": "Good morrow",
    "hi": "How now",
    "goodbye": "Fare thee well",
    "listen": "Hark",
    "wait": "Stay",
    "look": "Mark ye",
    "stop": "Soft you now",
    "hey": "What ho",
    "please": "Prithee",
    "indeed": "Forsooth",
    "truly": "Verily",
    "sir": "Gentle sir",
    "madam": "Mistress",
    "good morning": "Good morrow",
    "good evening": "Good e'en",
    "Indeed, sir,": "Marry, sir,",
    "Well, sir,": "Marry, sir,",
    "Honestly,": "Marry, sir,",
    "I beg you,": "Prithee,",
    "Goodness gracious,": "By'r lady,",
    "Wow,": "By'r lady,",
    "In truth,": "In sooth,",
    "Truly,": "In sooth,",
    "Certainly,": "Forsooth,",
    "Oh no,": "Alack,",
    "Unfortunately,": "Alack,",
    "Alas,": "Alack,",
    "Sadly,": "Alas,",
    "Hey, listen!": "Hark,",
    "Hear ye!": "Hark,",
    "Pay attention,": "Mark ye,",
    "Notice this,": "Mark ye,",
    "Listen up,": "Mark ye,",
    "Hold on a moment,": "Soft you now,",
    "Wait a minute,": "Soft you now,",
    "Quietly now,": "Soft you now,",
    "Wait,": "Stay,",
    "Hello!": "What ho,",
    "Hey there!": "What ho,",
    "What's up?": "What ho,",
    "What's happening?": "How now,",
    "What's this?": "How now,",
    "Well?": "How now,",
    "Silence!": "Peace,",
    "Be quiet!": "Peace,",
    "Quiet!": "Peace,",
    "My good sir,": "Gentle sir,",
    "Good morning,": "Good morrow,",
    "Good evening,": "God ye good den,",
    "Good day,": "God ye good den,",
}

def select_starter(text):
    text_lower = text.lower()
    for key_phrase, starter in starters_map.items():
        if text_lower.startswith(key_phrase):
            return starter
    return None

def postprocess_shakespeare(text, prefix_to_remove=None, add_starter=True):
    # Remove prefix if any
    if prefix_to_remove and text.startswith(prefix_to_remove):
        text = text[len(prefix_to_remove):].strip()

    # 1. Normalize contractions
    text = normalize_contractions(text)

    # 2. Phrase-level replacement
    text = phrase_replace(text, phrase_mapping)

    # 3. POS-aware substitution with spaCy
    doc = nlp(text)
    substituted_tokens = [pos_aware_substitution(token) for token in doc]

    # 4. Reconstruct text with spaces
    result = " ".join(substituted_tokens)

    # 5. Clean spacing before punctuation
    result = clean_text_spacing(result)

    # 6. Capitalize first letter

    def capitalize_first_alpha(text):
        for i, c in enumerate(text):
            if c.isalpha():
                return text[:i] + c.upper() + text[i+1:]
        return text  

    if result:
        result = capitalize_first_alpha(result)

    # 7. Add Shakespearean starter phrase if requested
    if add_starter:
        starter = select_starter(result)
        if starter:
            result = f"{starter}, {result}"

    return result

# Example usage
if __name__ == "__main__":
    sample_text = "Thank you for your help! I can't do this without you."
    print(postprocess_shakespeare(sample_text, add_starter=True))