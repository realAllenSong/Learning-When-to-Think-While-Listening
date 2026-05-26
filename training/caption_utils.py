"""
Caption cleanup utilities for reward-side chunk captions.

Step 4 captions are currently verbose. For reward computation we want shorter,
more local acoustic descriptions closer to the style of VGGSound labels without
destroying useful detail. These cleaners are intentionally light-touch and do
not modify the source JSONL on disk.
"""

import re
from typing import Callable


_PREFIX_PATTERNS = [
    r"^the primary sound is that of\s+",
    r"^the primary sound is\s+",
    r"^i heard\s+",
    r"^the audio contains\s+",
    r"^the clip contains\s+",
    r"^there is\s+",
    r"^there are\s+",
]

_FILLER_PATTERNS = [
    r"\bthe quality (?:or texture )?of the sound is\b",
    r"\bit has a quality of being\b",
    r"\bthe sound has a quality of being\b",
    r"\bthere (?:is|are) no noticeable changes? in intensity(?: over time)?\b",
    r"\bthere are no particular instruments used to produce this sound\b",
    r"\bthe intensity of the sound is\b",
    r"\bthe quality of the recording is\b",
    r"\bwhich lasts for approximately [0-9.]+ seconds?\b",
    r"\bfor about [0-9.]+ seconds?\b",
    r"\bfor approximately [0-9.]+ seconds?\b",
    r"\bthroughout the clip\b",
    r"\bwithin the clip\b",
]

_BOILERPLATE_PATTERNS = [
    r"^i am here to help you!?$",
    r"^i can answer .*",
    r"^i can help .*",
    r"^as an ai .*",
]

_HEDGING_PATTERNS = [
    r"\blikely\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\bappears to\b",
    r"\bseems to\b",
    r"\bmay be\b",
    r"\bmight be\b",
    r"\bsuggests that\b",
    r"\bindicates that\b",
    r"\bimplies that\b",
]

_QWEN3_FILLER_PATTERNS = [
    r"\bthe audio (?:appears|seems) to contain\b",
    r"\bthis audio (?:appears|seems) to contain\b",
    r"\bthe clip (?:appears|seems) to contain\b",
    r"\ba sense of\b",
    r"\bconveying a sense of\b",
    r"\bcreating a sense of\b",
    r"\bthe overall atmosphere is\b",
    r"\bthe emotional content is\b",
    r"\bindicative of\b",
    r"\bunderlying meaning\b",
]

_CANONICAL_MAP = {
    "silence": ["complete silence", "silence only", "mostly silence"],
    "background noise": ["ambient background noise", "low-level background noise", "steady background noise"],
    "breathing": ["breathing sounds", "sound of breathing"],
    "music": ["music playing", "instrumental music"],
    "speech": ["someone speaking", "a person speaking", "human speech"],
}

_QWEN3_EVENT_KEYWORDS = {
    "speech",
    "voice",
    "says",
    "say",
    "shouts",
    "shout",
    "yells",
    "yell",
    "scream",
    "screech",
    "shriek",
    "laugh",
    "laughter",
    "giggle",
    "giggling",
    "cheer",
    "cheering",
    "applause",
    "clap",
    "clapping",
    "music",
    "singing",
    "sing",
    "guitar",
    "piano",
    "drum",
    "drumming",
    "click",
    "clicking",
    "clack",
    "tick",
    "snap",
    "thud",
    "clunk",
    "clang",
    "bang",
    "beep",
    "buzz",
    "whoosh",
    "swoosh",
    "whine",
    "whir",
    "whirring",
    "hum",
    "humming",
    "grind",
    "grinding",
    "roar",
    "engine",
    "motor",
    "horn",
    "honk",
    "siren",
    "alarm",
    "bell",
    "ringing",
    "note",
    "string",
    "cello",
    "violin",
    "viola",
    "bass",
    "footsteps",
    "walking",
    "running",
    "dog",
    "bark",
    "barking",
    "cat",
    "purr",
    "meow",
    "meowing",
    "bird",
    "chirping",
    "chirp",
    "crow",
    "caw",
    "duck",
    "quack",
    "goose",
    "pig",
    "grunt",
    "snort",
    "rain",
    "wind",
    "water",
    "waves",
    "crowd",
    "crowds",
    "organ",
    "chord",
    "arpeggiated",
    "arpeggio",
    "melody",
    "snore",
    "snoring",
    "moo",
    "cow",
    "bellow",
    "guinea",
    "pigeon",
    "coo",
    "cooing",
    "sigh",
    "exhale",
    "exhalation",
    "breath",
    "tire",
    "skid",
    "squeal",
    "tearing",
    "torn",
    "paper",
    "zipper",
    "zip",
    "gurgling",
    "bubbling",
    "chewing",
    "crunching",
    "dance",
    "beat",
    "synth",
    "theme",
    "korean",
    "narrator",
}

_QWEN3_EVENT_NOUNS = [
    "pipe organ chord",
    "organ chord",
    "acoustic guitar chord",
    "guitar chord",
    "piano melody",
    "snoring",
    "cow moo",
    "guinea pig squeak",
    "pigeon coo",
    "tire squeal",
    "sigh",
    "metal door slam",
    "mechanical buzzing",
    "electric motor whine",
    "paper tearing",
    "table tennis",
    "dial tone",
    "gunshot",
    "firecracker",
    "firework",
    "toilet flush",
    "flush",
    "match",
    "whoosh",
    "swoosh",
    "howl",
    "caw",
    "crow",
    "duck",
    "quack",
    "goose",
    "honk",
    "purr",
    "pig",
    "snort",
    "grunt",
    "dog bark",
    "bark",
    "squeak",
    "click",
    "clack",
    "tick",
    "snap",
    "clink",
    "clang",
    "clank",
    "ping",
    "thud",
    "boom",
    "buzz",
    "buzzer",
    "hum",
    "tone",
    "hiss",
    "whine",
    "whirring",
    "whir",
    "screech",
    "scraping",
    "scrape",
    "strike",
    "impact",
    "rustling",
    "exhalation",
    "breath",
    "zipper",
    "zip",
    "bottle cap",
    "bowed string note",
    "cello note",
    "double bass note",
    "gurgling",
    "bubbling",
    "laughter",
    "theme music",
    "dance beat",
    "electronic dance music",
    "korean speech",
]

_QWEN3_META_KEYWORDS = {
    "audio",
    "clip",
    "recording",
    "environment",
    "space",
    "room",
    "indoors",
    "outdoors",
    "indoor",
    "outdoor",
    "high-fidelity",
    "fidelity",
    "mono",
    "stereo",
    "consumer-grade",
    "microphone",
    "noise",
    "floor",
    "equipment",
    "acoustically",
    "reverberant",
    "reverb",
    "close-miked",
    "centered",
    "soundscape",
    "atmosphere",
    "ambiance",
}

_QWEN3_INTRO_PATTERNS = [
    r"^the (?:audio clip|audio recording|audio|recording|clip)\s+(?:begins|opens|starts)\s+(?:abruptly\s+)?(?:with\s+)?",
    r"^the (?:audio clip|audio recording|audio|recording|clip)\s+is\s+(?:a|an)\s+[^.]{0,160}?\b(?:featuring|containing|marked by|with)\s+",
    r"^the (?:audio clip|audio recording|audio|recording|clip)\s+is\s+(?:a|an)\s+",
    r"^it\s+(?:begins|opens|starts)\s+(?:with\s+)?",
    r"^from the (?:very )?(?:first instant|start|very start)\s*,?\s*",
    r"^from the (?:outset|first moment|very beginning|moment the clip begins)\s*,?\s*",
    r"^from the very beginning to the abrupt end\s*,?\s*",
    r"^at the outset\s*,?\s*",
    r"^there is\s+",
    r"^there are\s+",
]

_QWEN3_META_PHRASES = [
    r"\bhigh-fidelity\b",
    r"\blow-fidelity\b",
    r"\bmono(?: recording)?\b",
    r"\bstereo(?: image)?\b",
    r"\bconsumer-grade\b",
    r"\bacoustically [a-z-]+\b",
    r"\bnoise floor\b",
    r"\bfield recording\b",
    r"\brecording device\b",
    r"\bclose-miked\b",
    r"\btechnically pristine\b",
    r"\bdominat(?:es|ing) the (?:recording|soundscape)\b",
    r"\bcapturing attention\b",
    r"\bsetting an atmosphere of [^,.;]+",
    r"\bcreating an atmosphere of [^,.;]+",
    r"\bjust over a second\b",
    r"\babout [0-9.]+ seconds?\b",
    r"\bfor [0-9.]+ seconds?\b",
    r"\blasting [^,.;]+\b",
    r"\bimmediately\b",
    r"\bsuddenly\b",
    r"\babruptly\b",
]

_QWEN3_DANGLING_ENDINGS = {
    "a",
    "an",
    "the",
    "of",
    "with",
    "and",
    "or",
    "its",
    "their",
    "his",
    "her",
    "this",
    "that",
    "these",
    "those",
    "indicating",
    "suggesting",
    "characterized",
    "accompanied",
    "followed",
    "produced",
    "delivered",
}

_QWEN3_LOW_VALUE_CAPTIONS = {
    "a single",
    "a loud",
    "a low-level",
    "a brief",
    "a sharp",
    "the environment is acoust",
    "the environment is",
    "this sound is a",
    "this sound is",
    "the sound is a",
    "the sound is",
    "in silence",
    "steady and unmodulated",
    "from the very start",
    "from the first instant",
    "from the very first instant",
}

_QWEN3_BAD_PREFIX_WORDS = {
    "this",
    "the",
    "a",
    "an",
    "in",
    "it",
    "following",
    "after",
    "there",
    "no",
    "with",
    "as",
    "at",
    "each",
    "its",
    "shortly",
    "captured",
    "capturing",
    "immersing",
    "presenting",
    "throughout",
    "beneath",
    "over",
}

_QWEN3_DISTINGUISHING_MODIFIERS = {
    "single",
    "double",
    "triple",
    "repeated",
    "rapid",
    "harsh",
    "distant",
    "sustained",
    "layered",
    "brief",
}

_QWEN3_COUNT_RULES = [
    (r"\b(?:three|triple|3)\b|\bsequence of three\b|\bthree distinct\b", "triple"),
    (r"\b(?:two|double|pair of)\b|\btwo rapid\b", "double"),
    (r"\b(?:single|solitary|lone)\b", "single"),
    (r"\b(?:series of|repeated|repeat(?:ed|ing)?|multiple|cluster of)\b", "repeated"),
]

_QWEN3_STYLE_RULES = [
    (r"\b(?:rapid|rapid-fire|staccato|quick succession)\b", "rapid"),
    (r"\b(?:harsh|raspy|nasal|guttural)\b", "harsh"),
    (r"\b(?:distant|far[- ]away|far away)\b", "distant"),
    (r"\b(?:sustained|drawn-out|drawn out|prolonged|elongated)\b", "sustained"),
    (r"\b(?:layered|overlapping|chorus)\b", "layered"),
]


def clean_chunk_caption(text: str, max_words: int = 18) -> str:
    """
    Convert verbose caption prose into a shorter reward-side acoustic note.

    The goal is not to mimic `train_swift.caption` exactly; it is to keep the
    caption chunk-local while removing boilerplate that makes the R_t judge too
    forgiving.
    """
    if not text:
        return ""

    cleaned = text.strip().strip('"').replace("\n", " ")
    cleaned = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    cleaned = cleaned.lower()

    for pattern in _BOILERPLATE_PATTERNS:
        if re.match(pattern, cleaned):
            return ""

    for pattern in _PREFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)

    for pattern in _FILLER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = cleaned.replace("specifically a ", "")
    cleaned = cleaned.replace("specifically an ", "")
    cleaned = cleaned.replace("specifically ", "")
    cleaned = cleaned.replace("indicating ", "")
    cleaned = cleaned.replace("suggesting ", "")
    cleaned = cleaned.replace("featuring ", "")
    cleaned = cleaned.replace("with an angry mood", "angry")
    cleaned = cleaned.replace("with a sad mood", "sad")
    cleaned = cleaned.replace("with a happy mood", "happy")
    cleaned = cleaned.replace("with a neutral mood", "neutral")
    cleaned = cleaned.replace("saying ", "says ")

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = cleaned.strip(" ,.;:-")

    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).strip(" ,.;:-")

    return cleaned


def clean_chunk_caption_qwen3(text: str, max_words: int = 16) -> str:
    """
    Stronger cleanup intended for Qwen3-Omni Captioner pilot outputs.

    Qwen3-Omni Captioner is designed for detailed low-hallucination captions.
    For reward-side chunk references we still want short, local, low-inference
    acoustic notes, so this variant strips more hedging and interpretive prose.
    """
    if not text:
        return ""

    normalized = text.strip().strip('"').replace("\n", " ").replace("—", ", ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    lowered = normalized.lower()
    used_salvage = False

    def _event_count_modifier(source: str) -> str | None:
        lowered_source = source.lower()
        for pattern, label in _QWEN3_COUNT_RULES:
            if re.search(pattern, lowered_source):
                return label
        return None

    def _event_style_modifier(source: str) -> str | None:
        lowered_source = source.lower()
        for pattern, label in _QWEN3_STYLE_RULES:
            if re.search(pattern, lowered_source):
                return label
        return None

    def _event_label_with_variation(
        source: str,
        singular: str,
        *,
        plural: str | None = None,
        allow_count: bool = True,
        allow_style: bool = True,
    ) -> str:
        count_modifier = _event_count_modifier(source) if allow_count else None
        style_modifier = _event_style_modifier(source) if allow_style else None
        plural_label = plural or singular

        if count_modifier == "single":
            return f"single {singular}"
        if count_modifier in {"double", "triple", "repeated"}:
            return f"{count_modifier} {plural_label}"
        if style_modifier:
            return f"{style_modifier} {plural_label}"
        return singular

    def _has_distinguishing_modifier(cleaned_text: str) -> bool:
        words = cleaned_text.lower().split()
        if not words:
            return False
        if words[0] in _QWEN3_DISTINGUISHING_MODIFIERS:
            return True
        return any(
            token in cleaned_text.lower()
            for token in ["high-pitched", "low-frequency", "multi-", "double ", "triple "]
        )

    def _strip_negative_context(source: str) -> str:
        cleaned_source = source
        negative_patterns = [
            r"\bno (?:speech|spoken words|voice|voices|music|lyrics|ambient noise|background noise|hum|hiss|reverberation|echo|environmental sounds?)\b[^.]*",
            r"\bwithout (?:speech|spoken words|voice|voices|music|lyrics|ambient noise|background noise|hum|hiss|reverberation|echo|environmental sounds?)\b[^.]*",
            r"\babsence of (?:speech|spoken words|voice|voices|music|lyrics|ambient noise|background noise|hum|hiss|reverberation|echo|environmental sounds?)\b[^.]*",
            r"\bdevoid of (?:speech|spoken words|voice|voices|music|lyrics|ambient noise|background noise|hum|hiss|reverberation|echo|environmental sounds?)\b[^.]*",
            r"\bfree from (?:speech|spoken words|voice|voices|music|lyrics|ambient noise|background noise|hum|hiss|reverberation|echo|environmental sounds?)\b[^.]*",
            r"\b(?:lack|lacks) of (?:any )?(?:speech|spoken words|voice|voices|music|lyrics|ambient noise|background noise|hum|hiss|reverberation|echo|environmental sounds?)\b[^.]*",
        ]
        for pattern in negative_patterns:
            cleaned_source = re.sub(pattern, " ", cleaned_source)
        return re.sub(r"\s+", " ", cleaned_source).strip()

    def _salvage_event_phrase(source: str) -> str:
        candidate = _strip_negative_context(source.lower())
        for pattern in _QWEN3_INTRO_PATTERNS:
            candidate = re.sub(pattern, "", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip(" ,.;:-")

        quote_match = re.search(r"[\"“]([^\"”]{1,40})[\"”]", source)
        if quote_match:
            quoted = quote_match.group(1).strip().lower().strip(" ,.;:!?-")
            speech_context = source.lower()
            has_positive_speech = bool(
                re.search(r"\b(?:says?|shouts?|yells?|utters?|speaks?|spoken by|narrator|male voice|female voice|human voice|speaker|speaking)\b", speech_context)
            )
            has_negative_speech = bool(re.search(r"\bno (?:speech|spoken words|voice|voices)\b", speech_context))
            if quoted and has_positive_speech and not has_negative_speech:
                short_quoted = " ".join(quoted.split()[:6]).strip(" ,.;:-")
                if "male" in speech_context and any(token in speech_context for token in {"speech", "voice", "narrator", "sentence", "speaker", "speaking"}):
                    return f"male says {short_quoted}"
                if "female" in speech_context and any(token in speech_context for token in {"speech", "voice", "vocal", "speaker", "speaking"}):
                    return f"female says {short_quoted}"
                if any(token in speech_context for token in {"speech", "voice", "narrator", "sentence", "speaker", "speaking"}):
                    return f"speech says {short_quoted}"

        direct_rules = [
            (r"\bpipe organ\b.*\bchord\b|\borgan\b.*\bchord\b", "pipe organ chord"),
            (r"\bacoustic guitar\b.*\b(?:chord|arpeggi(?:o|ated)|melody)\b|\bguitar\b.*\b(?:chord|arpeggi(?:o|ated)|melody)\b", "acoustic guitar chord"),
            (r"\bpiano\b.*\b(?:melody|phrase|notes?)\b", "piano melody"),
            (r"\b(?:snore|snoring|snorts? in sleep)\b", "snoring"),
            (r"\b(?:sigh|exhalation|forceful exhalation|deep exhalation|breath(?:ing)?)\b", "sigh"),
            (r"\bcow\b.*\b(?:moo|bellow|lowing)\b|\b(?:moo|bellow)\b.*\bcow\b", "cow moo"),
            (r"\bguinea pig\b.*\b(?:squeak|squealing|chirp|vocalization)\b|\bguinea pig\b", "guinea pig squeak"),
            (r"\bpigeon\b.*\b(?:coo|cooing|wing|flutter|vocalization)\b|\bpigeons?\b", "pigeon coo"),
            (r"\bbird\b.*\b(?:call|song|chirp|chirping|trill|tweet|tweeting|coo|cooing|sing|singing)\b|\b(?:birdsong|bird call)\b", "bird call"),
            (r"\b(?:tire|tyre)\b.*\b(?:squeal|skid|skidding|losing traction)\b|\bbrake squeal\b", "tire squeal"),
            (r"\b(?:metal|metallic)\b.*\b(?:door|gate)\b.*\b(?:slam|shut|close|closing|lock|locked)\b", "metal door slam"),
            (r"\b(?:metal|metallic)\b.*\b(?:impact|clang|clatter|rattle|clink|strik(?:e|ing)|dropped)\b", "metallic impact"),
            (r"\b(?:metal|metallic)\b.*\b(?:screech|squeal)\b", "metallic screech"),
            (
                r"\b(?:muscle car|sports car|supercar|engine)\b.*\b(?:rev|revving|launch|roar|rumble|acceleration)\b",
                _event_label_with_variation(candidate, "engine roar", allow_count=False),
            ),
            (r"\b(?:electric motor|rotary tool|razor|small powered device)\b.*\b(?:whine|whir|buzz|buzzing)\b", "electric motor whine"),
            (r"\b(?:mechanical|electric)\b.*\b(?:buzz|buzzing|whine|whir|whirring)\b", "mechanical buzzing"),
            (r"\b(?:mechanical|tire|brake)\b.*\b(?:squeal|screech)\b", "mechanical squeal"),
            (r"\b(?:engine|motor|diesel|industrial motor)\b.*\b(?:steady|unchanging|idle|idling|constant|hum|rumble)\b", "engine hum"),
            (r"\b(?:dog|canine)\b.*\b(?:bark|yelp|yelps|howl|vocalization)\b|\bcanine vocalization\b", "dog bark"),
            (r"\bgrowl(?:s|ing)?\b|\bguttural growls?\b", "animal growl"),
            (r"\bbull\b.*\bbellow\b|\bbellow\b.*\bbull\b", "bull bellow"),
            (r"\b(?:fly|housefly|insect)\b.*\b(?:buzz|buzzing|flutter)\b", "fly buzz"),
            (r"\b(?:jet|fighter jet|aircraft|turbofan)\b.*\b(?:pass|passes|passing|flyover|doppler|whine|roar)\b", "jet flyover"),
            (r"\bhelicopter\b|\brotor\b.*\b(?:roar|chop|blade)\b", "helicopter noise"),
            (r"\b(?:sheet of )?paper\b.*\b(?:tear(?:ing|s)?|torn|ripp?ed|rip)\b", "paper tearing"),
            (r"\bkorean\b.*\b(?:sentence|speech|narrator|voice)\b", "korean speech"),
            (r"\b(?:electronic dance music|dance music|house music)\b", "electronic dance music"),
            (r"\b(?:dance beat|four-on-the-floor|kick drum|synth bassline|hi-hats?)\b", "electronic dance beat"),
            (r"\bwhoosh\b|\bswoosh\b", "synthetic whoosh"),
            (
                r"\b(?:car|vehicle|truck|sedan|suv|bus)\b.*\bhorn\b|\b(?:car horn|vehicle horn|horn blast)\b",
                "vehicle horn",
            ),
            (
                r"\b(?:aerosol|spray can|pressurized)\b.*\b(?:hiss|spray|expelled)\b|\baerosol hiss\b",
                "aerosol hiss",
            ),
            (
                r"\b(?:waterfall|rapids|rushing water|torrent|cascad(?:e|ing))\b.*\b(?:roar|rush|hiss|spray)\b",
                "rushing water",
            ),
            (r"\b(?:metallic|mechanical)\b.*\b(?:tick|clack|click|snap)\b", "metallic click"),
            (r"\b(?:tick|clack)\b", "metallic tick"),
            (r"\bgoose\b|\bhonk\b", "goose honks"),
            (r"\bcat\b.*\bpurr\b|\bpurr\b", "cat purr"),
            (r"\b(?:plastic )?(?:bottle|jar)\b.*\b(?:cap|snap|twist|opening)\b", "plastic bottle cap snap"),
            (r"\bzipper\b|\bzip\b", "zipper zip"),
            (r"\b(?:bowed string|cello|double bass|upright bass|viola|violin)\b.*\bnote\b", "bowed string note"),
            (r"\b(?:sustained|single)\s+note\b.*\b(?:cello|double bass|upright bass|viola|violin|bowed string)\b", "bowed string note"),
            (r"\bgurgling\b|\bbubbling\b", "liquid gurgling"),
            (r"\b(?:chew|chewing|crunching|squelching|bite|biting)\b", "wet crunching chew"),
            (r"\b(?:child|toddler|preschooler)\b.*\blaugh", "child laughter"),
            (r"\blaugh(?:ter|ing)?\b|\bgiggl(?:e|es|ing)\b", "laughter"),
            (r"\baddams family\b|\btheme music\b|\btheme of\b", "theme music"),
            (r"\bduck\b|\bquack\b", "duck quacks"),
            (r"\bhand clap\b|\b(?:single|solitary)\s+hand clap\b", "hand clap"),
            (r"\b(?:single|sharp|percussive)\s+sound\b.*\b(?:metallic|click|tick|clack)\b", "metallic click"),
            (r"\bpig\b.*\b(?:snort|sniff|grunt|gurgl|squelch)\b|\b(?:snort|grunt)\b.*\bpig\b", "pig snorts and grunts"),
            (
                r"\bcrow\b.*\bcaw\b|\bcaw(?:s|ing)?\b.*\bcrow\b|\bcrow(?:s|ing)?\b|\bcaw(?:s|ing)?\b",
                _event_label_with_variation(candidate, "crow caw", plural="crow caws"),
            ),
            (r"\brooster\b", "rooster crow"),
            (
                r"\b(?:beep|beeps|buzzer|buzzers|alert tone|notification tone|alarm signal)\b",
                _event_label_with_variation(candidate, "electronic beep", plural="electronic beeps"),
            ),
        ]
        for pattern, label in direct_rules:
            if re.search(pattern, candidate):
                return label

        for noun in _QWEN3_EVENT_NOUNS:
            noun_pattern = re.escape(noun)
            match = re.search(
                rf"((?:[a-z0-9'/-]+\s+){{0,4}}{noun_pattern}(?:\s+[a-z0-9'/-]+){{0,4}})",
                candidate,
            )
            if match:
                phrase = match.group(1).strip(" ,.;:-")
                phrase = re.sub(r"^(?:a|an|the)\s+", "", phrase)
                phrase = re.sub(r"\b(?:that|which)\b.*$", "", phrase)
                words = phrase.split()
                while words and words[-1] in _QWEN3_DANGLING_ENDINGS:
                    words.pop()
                phrase = " ".join(words).strip(" ,.;:-")
                if phrase and phrase not in _QWEN3_LOW_VALUE_CAPTIONS:
                    return phrase

        generic = candidate
        generic = re.sub(r"[\"“”]", "", generic)
        generic = re.sub(r"\b(?:this|that|which|whose|with|without|throughout|overall|summary)\b", " ", generic)
        generic = re.sub(
            r"\b(?:recording|audio|clip|environment|studio|quality|fidelity|signal|context|purpose|production|setting|microphone|equipment|space)\b",
            " ",
            generic,
        )
        generic = re.sub(r"\s+", " ", generic).strip(" ,.;:-")
        clauses = [seg.strip(" ,.;:-") for seg in re.split(r"[.;:]", generic) if seg.strip(" ,.;:-")]
        if clauses:
            best_clause = max(clauses[:4], key=_score_sentence)
            clause_words = best_clause.split()
            if clause_words:
                trimmed = " ".join(clause_words[: min(8, len(clause_words))]).strip(" ,.;:-")
                trimmed = re.sub(r"^(?:a|an|the)\s+", "", trimmed)
                if trimmed and trimmed not in _QWEN3_LOW_VALUE_CAPTIONS:
                    return trimmed
        return ""

    def _canonicalize_event_label(cleaned_text: str, source_text: str) -> str:
        if _has_distinguishing_modifier(cleaned_text):
            return cleaned_text
        combined = f"{cleaned_text.lower()} || {source_text.lower()}"
        if "kookaburra" in combined:
            return "bird call"
        label_rules = [
            (r"\bpaper\b.*\b(?:tear(?:ing|s)?|rip|ripp?ed|torn)\b", "paper tearing"),
            (r"\btable tennis\b|\bpaddle striking a ball\b", "table tennis hit"),
            (r"\b(?:male says|female says|speech says)\b", cleaned_text.lower()),
            (r"\bkorean\b.*\b(?:speech|sentence|narrator|voice)\b", "korean speech"),
            (r"\bsiren\b", "siren"),
            (r"\b(?:electronic dance music|dance music|house music)\b", "electronic dance music"),
            (r"\btheme music\b|\baddams family\b", "theme music"),
            (r"\bcat\b.*\bmeow\b|\bmeow\b", "cat meow"),
            (r"\bcat\b.*\bpurr\b|\bpurr\b", "cat purr"),
            (r"\bdog\b.*\b(?:bark|yelp|howl)\b|\b(?:dog bark|dog yelp)\b", "dog bark"),
            (
                r"\bcrow\b.*\bcaw\b|\bcaw(?:s|ing)?\b.*\bcrow\b|\bcrow caw\b",
                _event_label_with_variation(source_text, "crow caw", plural="crow caws"),
            ),
            (r"\brooster\b", "rooster crow"),
            (r"\bduck\b|\bquack\b", "duck quacks"),
            (r"\bgoose\b|\bhonk\b", "goose honks"),
            (r"\bcamel\b", "camel call"),
            (r"\bbull\b.*\bbellow\b|\bbellow\b.*\bbull\b", "bull bellow"),
            (r"\b(?:fly|housefly|insect)\b.*\b(?:buzz|buzzing|flutter)\b", "fly buzz"),
            (
                r"\b(?:bird|pigeon|dove|owl|sparrow|robin|gull|seagull)\b.*\b(?:call|song|chirp|chirping|tweet|tweeting|coo|cooing|sing|singing|cry|vocalization)\b",
                _event_label_with_variation(source_text, "bird call", plural="bird calls"),
            ),
            (r"\b(?:jet|fighter jet|aircraft|turbofan)\b.*\b(?:pass|passes|passing|flyover|doppler|whine|roar)\b", "jet flyover"),
            (r"\bhelicopter\b|\brotor\b.*\b(?:roar|chop|blade)\b", "helicopter noise"),
            (r"\b(?:engine|motor|diesel|industrial motor)\b.*\b(?:steady|unchanging|idle|idling|constant|hum|rumble)\b", "engine hum"),
            (
                r"\b(?:car|vehicle|truck|sedan|suv|bus)\b.*\bhorn\b|\b(?:car horn|vehicle horn|horn blast)\b",
                "vehicle horn",
            ),
            (r"\b(?:train|subway|locomotive|railroad|railcar|rail)\b.*\bhorn\b", "train horn"),
            (r"\b(?:train|subway|locomotive|railroad|railcar|rail)\b.*\b(?:rumble|rattle|track|wheel|passing|approach)\b", "train rumble"),
            (r"\bwhoosh\b|\bswoosh\b", "synthetic whoosh"),
            (
                r"\b(?:aerosol|spray can|pressurized)\b.*\b(?:hiss|spray|expelled)\b|\baerosol hiss\b",
                "aerosol hiss",
            ),
            (
                r"\b(?:waterfall|rapids|rushing water|torrent|cascad(?:e|ing))\b.*\b(?:roar|rush|hiss|spray)\b",
                "rushing water",
            ),
            (r"\b(?:toilet|flush)\b", "toilet flush"),
            (r"\b(?:water|spray|splash)\b.*\b(?:gurgling|bubbling|rush|splash)\b|\bwater splash\b", "water splash"),
            (r"\b(?:gunshot|gunfire|firearm|rifle|submachine gun|machine gun)\b", "gunfire"),
            (r"\b(?:hand clap|clap)\b", "hand clap"),
            (r"\bpig\b.*\b(?:snort|grunt|sniff)\b|\b(?:pig snorts and grunts)\b", "pig snorts and grunts"),
            (r"\b(?:cello|double bass|upright bass|viola|violin|bowed string)\b.*\bnote\b|\bbowed string note\b", "bowed string note"),
            (r"\b(?:zipper|zip)\b", "zipper zip"),
            (r"\b(?:bottle|jar)\b.*\b(?:cap|snap|twist|opening)\b", "plastic bottle cap snap"),
            (r"\b(?:crunching|chewing|squelching|bite|biting)\b", "wet crunching chew"),
            (
                r"\b(?:beep|beeps|buzzer|buzzers|alert tone|notification tone|alarm signal)\b",
                _event_label_with_variation(source_text, "electronic beep", plural="electronic beeps"),
            ),
        ]
        for pattern, label in label_rules:
            if re.search(pattern, combined):
                return label
        return cleaned_text

    def _infer_primary_non_speech_label(source_text: str) -> str | None:
        source = _strip_negative_context(source_text.lower())
        rules = [
            (r"\b(?:toilet|flush)\b", "toilet flush"),
            (
                r"\b(?:shower|pour(?:ing|ed)?|droplets|gurgling|bubbling|water being poured|water displaces air|high-pressure jet)\b",
                "water splash",
            ),
            (
                r"\b(?:waterfall|rapids|rushing water|torrent|cascad(?:e|ing)|water feature|large body of water|massive volume of water)\b",
                "rushing water",
            ),
            (r"\bhelicopter\b|\brotor\b.*\b(?:roar|chop|blade)\b", "helicopter noise"),
            (r"\b(?:jet|fighter jet|aircraft|turbofan)\b.*\b(?:pass|passes|passing|flyover|doppler|whine|roar)\b", "jet flyover"),
            (r"\bsiren\b", "siren"),
            (r"\b(?:train|subway|locomotive|railroad|railcar|rail)\b.*\b(?:rumble|rattle|track|wheel|passing|approach|screech)\b", "train rumble"),
            (r"\b(?:car|vehicle|truck|sedan|suv|bus)\b.*\bhorn\b|\b(?:car horn|vehicle horn|horn blast)\b", "vehicle horn"),
            (
                r"\b(?:muscle car|sports car|supercar|motorcycle|engine)\b.*\b(?:rev|revving|launch|roar|rumble|acceleration|high rpm|full throttle)\b",
                "engine roar",
            ),
            (r"\b(?:engine|motor|diesel|industrial motor|vehicle)\b.*\b(?:steady|unchanging|idle|idling|constant|hum|rumble)\b", "engine hum"),
            (r"\bbull\b.*\bbellow\b|\bbellow\b.*\bbull\b", "bull bellow"),
            (r"\b(?:fly|housefly|insect|bumblebee|bee)\b.*\b(?:buzz|buzzing|flutter|drone)\b", "fly buzz"),
            (r"\bcrow\b.*\bcaw\b|\bcaw(?:s|ing)?\b.*\bcrow\b", "crow caw"),
            (r"\bgoose\b|\bhonk\b", "goose honks"),
            (r"\b(?:bird|pigeon|dove|owl|sparrow|robin|gull|seagull)\b.*\b(?:call|song|chirp|chirping|tweet|tweeting|coo|cooing|sing|singing|cry|vocalization)\b", "bird call"),
            (r"\b(?:dog|canine)\b.*\b(?:bark|yelp|howl|vocalization)\b", "dog bark"),
            (r"\b(?:metal|metallic)\b.*\b(?:impact|clang|clatter|rattle|clink|strike|striking|dropped)\b", "metallic impact"),
            (r"\b(?:metal|metallic|mechanical)\b.*\b(?:tick|clack|click|snap)\b", "metallic click"),
        ]
        for pattern, label in rules:
            if re.search(pattern, source):
                return label
        return None

    def _speech_is_secondary(source_text: str) -> bool:
        source = source_text.lower()
        secondary_markers = [
            "brief",
            "faint",
            "masked by",
            "heavily masked",
            "muffled",
            "indistinct",
            "unintelligible",
            "single word",
            "single phrase",
            "at approximately",
            "at the very beginning",
            "fleeting",
            "distant",
            "barely audible",
        ]
        return any(marker in source for marker in secondary_markers)

    for canonical, variants in _CANONICAL_MAP.items():
        if lowered == canonical or lowered in variants:
            return canonical

    # Prefer the sentence that contains actual sound-event evidence rather than
    # recording metadata. Raw Qwen3 captions often start with prose about the
    # recording setup and only later describe the event.
    sentences = [
        s.strip(" ,.;:-")
        for s in re.split(r"(?<=[.!?])\s+", normalized)
        if s.strip(" ,.;:-")
    ]
    if not sentences:
        return ""

    def _score_sentence(sentence: str) -> tuple[int, int]:
        sent = sentence.lower()
        event_hits = sum(1 for word in _QWEN3_EVENT_KEYWORDS if word in sent)
        meta_hits = sum(1 for word in _QWEN3_META_KEYWORDS if word in sent)
        return (event_hits - meta_hits, event_hits)

    best_sentence = _strip_negative_context(max(sentences[:3], key=_score_sentence))
    cleaned = best_sentence.lower()

    for pattern in _QWEN3_INTRO_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    for pattern in _BOILERPLATE_PATTERNS:
        if re.match(pattern, cleaned):
            return ""
    for pattern in _QWEN3_FILLER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    for pattern in _HEDGING_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)
    for pattern in _QWEN3_META_PHRASES:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = cleaned.replace("it sounds like", "")
    cleaned = cleaned.replace("can be heard", "")
    cleaned = cleaned.replace("can be perceived", "")
    cleaned = cleaned.replace("audio of", "")
    cleaned = cleaned.replace("sound of", "")
    cleaned = cleaned.replace("a sound of", "")
    cleaned = cleaned.replace("of a domestic guinea pig", "guinea pig squeak")
    cleaned = cleaned.replace("an audio clip of", "")
    cleaned = cleaned.replace("an audio recording of", "")
    cleaned = cleaned.replace("the clip features", "")
    cleaned = cleaned.replace("the clip includes", "")
    cleaned = cleaned.replace("audio captures", "")
    cleaned = cleaned.replace("the dominant element is", "")
    cleaned = cleaned.replace("the main sound is", "")
    cleaned = cleaned.replace("the primary sound is", "")
    cleaned = cleaned.replace("characterized by", "")
    cleaned = cleaned.replace("marked by", "")
    cleaned = cleaned.replace("consisting of", "")
    cleaned = cleaned.replace("followed by", ", ")
    cleaned = cleaned.replace("accompanied by", ", ")
    cleaned = re.sub(r"^(?:this|the) sound is\s+(?:a|an)?\s*", "", cleaned)
    cleaned = re.sub(r"^(?:this|the)\s+(?:whine|hum|buzz|buzzer|noise)\s*,\s*", "", cleaned)
    cleaned = re.sub(r"^(?:the\s+)?abrupt onset of\s+", "", cleaned)
    cleaned = re.sub(r"^[^:]{0,80}\bevent:\s*", "", cleaned)
    cleaned = re.sub(r"^(?:these noises?|this sound|these sounds?)\s+(?:suggest|suggests|indicate|indicates|imply|implies)\s+that\s+", "", cleaned)
    cleaned = re.sub(r"^(?:and|or)\s+", "", cleaned)
    cleaned = re.sub(r"^with\s+", "", cleaned)
    cleaned = re.sub(r"^(?:total\s+)?lack of (?:any\s+)?", "", cleaned)
    cleaned = re.sub(r"\bthat (?:dominates|fills|overwhelms|saturates)[^,.;]*$", "", cleaned)
    cleaned = re.sub(r"\b(?:immediately )?(?:filling|saturating|establishing)[^,.;]*$", "", cleaned)
    cleaned = re.sub(r"\bin an enthusiastic[^,.;]*$", "", cleaned)
    cleaned = re.sub(r"\bjust as suddenly\b.*$", "", cleaned)
    cleaned = re.sub(r"\bfrom the (?:very )?(?:first instant|start|very start)\b", "", cleaned)
    cleaned = re.sub(r"\bfrom the (?:outset|first moment|very beginning|moment the clip begins)\b", "", cleaned)
    cleaned = re.sub(r"\bat the outset\b", "", cleaned)
    cleaned = cleaned.replace("the word ", "")
    cleaned = re.sub(r"\s*!\s*", " ", cleaned)

    # Speech-oriented compression.
    cleaned = re.sub(
        r"\b(?:a |an )?(male|female|child|woman|man) voice[^.]*?\b(?:says?|shouts?|yells?|utters?)\s+[\"“]?([a-z0-9' -]{1,40})[\"”]?",
        r"\1 says \2",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(?:someone|a person|person)[^.]*?\b(?:says?|shouts?|yells?|utters?)\s+[\"“]?([a-z0-9' -]{1,40})[\"”]?",
        r"speech says \1",
        cleaned,
    )
    cleaned = re.sub(r"\bmale voice\b", "male speech", cleaned)
    cleaned = re.sub(r"\bfemale voice\b", "female speech", cleaned)
    cleaned = re.sub(r"\bcanine vocalization\b", "dog bark", cleaned)

    # Remove trailing explanatory clauses that are usually not chunk-local evidence.
    cleaned = re.sub(r",\s*(?:indicating|suggesting|implying|which|that)\b.*$", "", cleaned)
    cleaned = re.sub(r"\b(?:in a|within a|inside a)\s+[^,.;]{0,80}\b(?:environment|space|room|hall|setting)\b", "", cleaned)
    cleaned = re.sub(r"\bwith no [^,.;]+", "", cleaned)
    cleaned = re.sub(r"\bthere (?:is|are) no [^,.;]+", "", cleaned)
    cleaned = re.sub(r"\b(?:total\s+)?lack of (?:any\s+)?[^,.;]+", "", cleaned)

    # Keep only the most event-like comma segments.
    segments = [seg.strip(" ,.;:-") for seg in re.split(r"\s*,\s*", cleaned) if seg.strip(" ,.;:-")]
    kept_segments = []
    for segment in segments:
        sent_score = _score_sentence(segment)
        if sent_score[1] > 0 or not kept_segments:
            kept_segments.append(segment)
        if len(kept_segments) >= 2:
            break
    cleaned = ", ".join(kept_segments)

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = cleaned.strip(" ,.;:-")

    words = cleaned.split()
    while words and words[-1] in _QWEN3_DANGLING_ENDINGS:
        words.pop()
    cleaned = " ".join(words).strip(" ,.;:-")

    if not cleaned:
        cleaned = _salvage_event_phrase(normalized)
        if not cleaned:
            return ""
        used_salvage = True

    lowered = cleaned.lower()
    for canonical, variants in _CANONICAL_MAP.items():
        if lowered == canonical or lowered in variants:
            return canonical
    if lowered in _QWEN3_LOW_VALUE_CAPTIONS:
        rescued = _salvage_event_phrase(normalized)
        if rescued:
            cleaned = rescued
            lowered = cleaned.lower()
            used_salvage = True
        else:
            return ""

    bad_start = bool(cleaned.split()) and cleaned.split()[0].lower().rstrip(",") in _QWEN3_BAD_PREFIX_WORDS
    bad_structure = bool(
        re.match(
            r"^(?:the|this)\s+(?:sound|event|sequence|recording|clip|audio|vocalization|tone|rumble|splash|engine|meow|bark|grunt|laugh|scream)\b",
            lowered,
        )
    )
    explanatory = any(
        token in lowered
        for token in [
            " begins with ",
            " begins ",
            " opens with ",
            " opens ",
            " starts with ",
            " starts ",
            " is ",
            " are ",
            " features ",
            " consists ",
            " dominated by ",
            " immersive ",
            " indicating ",
            " suggesting ",
            " evoc",
        ]
    )
    if bad_start or bad_structure or explanatory:
        rescued = _salvage_event_phrase(normalized)
        if rescued:
            cleaned = rescued
            lowered = cleaned.lower()
            used_salvage = True

    residual_fragment = bool(
        lowered.startswith(
            (
                "as ",
                "for ",
                "from the ",
                "from ",
                "dominated by ",
                "steady and ",
                "steady ",
                "is captured as it ",
                "captured as it ",
            )
        )
        or any(
            phrase in lowered
            for phrase in [
                "for the act of",
                "as it ",
                "just as ",
                "steady and unmodulated",
                "steady and unchanging throughout",
                "engine’s deep",
                "engine's deep",
                "aggressive engine note",
                "rumbling engine note",
                "resonant bellow",
                "enveloping a large volume of water in motion",
                "grinding noise dominates",
            ]
        )
    )
    if residual_fragment:
        repaired = _canonicalize_event_label(cleaned, normalized)
        if repaired and repaired != cleaned:
            cleaned = repaired
            lowered = cleaned.lower()
        else:
            cleaned = re.sub(r"^(?:is\s+)?captured as it\s+", "", cleaned)
            cleaned = re.sub(r"\bas it passes\b", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")
            lowered = cleaned.lower()

    speech_like = bool(re.match(r"^(?:male|female|speech) says\b", lowered))
    artifact_like = lowered in {
        "electronic beep",
        "single electronic beep",
        "double electronic beeps",
        "triple electronic beeps",
        "repeated electronic beeps",
        "ping and distortion",
        "digital artifacts",
    }
    dominant_non_speech = _infer_primary_non_speech_label(normalized)
    if dominant_non_speech:
        if artifact_like:
            cleaned = dominant_non_speech
            lowered = cleaned.lower()
        elif speech_like and _speech_is_secondary(normalized):
            cleaned = dominant_non_speech
            lowered = cleaned.lower()

    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).strip(" ,.;:-")
        words = cleaned.split()
        while words and words[-1] in _QWEN3_DANGLING_ENDINGS:
            words.pop()
        cleaned = " ".join(words).strip(" ,.;:-")
    if cleaned.lower() == "from the outset":
        rescued = _salvage_event_phrase(normalized)
        if rescued and rescued.lower() != cleaned.lower():
            cleaned = rescued
            lowered = cleaned.lower()
    if cleaned.lower() in _QWEN3_LOW_VALUE_CAPTIONS:
        rescued = _salvage_event_phrase(normalized)
        if rescued:
            return rescued
        return ""
    if any(
        phrase in cleaned.lower()
        for phrase in ["for the act of", "as it ", "just as ", "steady and unmodulated", "from the very start", "from the first instant"]
    ):
        rescued = _salvage_event_phrase(normalized)
        if rescued and rescued.lower() != cleaned.lower():
            cleaned = rescued
            lowered = cleaned.lower()
        elif cleaned.lower() in _QWEN3_LOW_VALUE_CAPTIONS:
            return ""

    if used_salvage and len(cleaned.split()) <= 4:
        cleaned = _canonicalize_event_label(cleaned, normalized)
    return cleaned.strip(" ,.;:-")


def resolve_caption_cleaner(name: str) -> Callable[[str], str]:
    cleaner = (name or "legacy").strip().lower()
    if cleaner in {"none", "raw"}:
        return lambda text: (text or "").strip().strip('"')
    if cleaner in {"legacy", "qwen2", "qwen2-audio"}:
        return clean_chunk_caption
    if cleaner in {"qwen3", "qwen3-captioner", "captioner"}:
        return clean_chunk_caption_qwen3
    raise ValueError(f"Unsupported caption cleaner: {name}")
