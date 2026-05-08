"""
prompts.py — All 8 prompt conditions for the HAM10000 pilot.

Each condition is a function that returns a `messages` list in the format
MedGemma expects. This format is the standard transformers chat-template
shape: a list of dicts with `role` and `content`, where `content` is itself
a list of typed parts (text/image).

For the pilot we only use P1 and P4 (most-different conditions).
The full 8 are here so you don't have to rewrite later.

Design: prompts are composed from shared building blocks below so the
factorial axes (shots × role × CoT) stay strictly orthogonal — only the
manipulated axis differs across conditions.
"""

# Class vocabulary used in every prompt. Keep these consistent everywhere.
CLASSES_DESC = """- mel:   melanoma
- nv:    melanocytic nevus
- bcc:   basal cell carcinoma
- akiec: actinic keratosis or intraepithelial carcinoma
- bkl:   benign keratosis (seborrheic keratosis, solar lentigo)
- df:    dermatofibroma
- vasc:  vascular lesion (hemangioma, angioma)"""

CLASSES_SHORT = "mel, nv, bcc, akiec, bkl, df, vasc"


# ---------- Shared building blocks (identical across all 8 conditions) ----------

# Constraints applied uniformly so the only thing varying across conditions
# is the manipulated axis (role / CoT / shots).
CONSTRAINTS_BLOCK = """Constraints:
- Pick exactly one class. Do not refuse.
- Avoid defaulting to a single class without visual evidence.
- Use only the codes above. No synonyms in the final answer."""

# CoT-only addition: explicit reasoning steps + output-template change.
COT_STEPS_BLOCK = """Reason through these steps explicitly:
1. Asymmetry / Border / Color / Structures (the ABCD evaluation)
2. Top 2 candidates with one reason each
3. Final commitment"""

# Output formats — direct gets just the answer line; CoT gets a reasoning line first.
OUTPUT_DIRECT = """Output exactly this format and nothing else:

Final answer: <class_code>"""

OUTPUT_COT = """Output exactly this format and nothing else:

Reasoning: <your 2-3 sentence analysis>
Final answer: <class_code>"""

# Role-only addition: system message identifying the assistant as a dermatologist.
ROLE_SYSTEM = ("You are an experienced dermatologist analyzing a dermoscopic image "
               "for a research benchmark.")


def _user_msg(image, instruction_text):
    """A single user turn with one image and one block of instructions."""
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction_text},
        ],
    }


def _system_msg(text):
    return {"role": "system", "content": [{"type": "text", "text": text}]}


def _build_instruction(cot: bool) -> str:
    """Compose the user-message instruction text from shared blocks."""
    header = f"""Classify this dermoscopic image into exactly one of these classes:

{CLASSES_DESC}

{CONSTRAINTS_BLOCK}"""
    if cot:
        return f"{header}\n\n{COT_STEPS_BLOCK}\n\n{OUTPUT_COT}"
    return f"{header}\n\n{OUTPUT_DIRECT}"


# ---------- The 8 conditions ----------
# Naming convention: P<shots><role><reasoning>
#   shots:    0 = zero-shot, F = few-shot
#   role:     N = no role,   R = role
#   reasoning: D = direct,   C = CoT
#
# So P1 = P0ND  (zero-shot, no role, direct)
#    P4 = P0RC  (zero-shot, role, CoT)
#    P8 = PFRC  (few-shot, role, CoT)


# ----- P1: zero-shot, no role, direct -----
def p1_zero_norole_direct(image):
    return [_user_msg(image, _build_instruction(cot=False))]


# ----- P2: zero-shot, no role, CoT -----
def p2_zero_norole_cot(image):
    return [_user_msg(image, _build_instruction(cot=True))]


# ----- P3: zero-shot, role, direct -----
def p3_zero_role_direct(image):
    return [_system_msg(ROLE_SYSTEM), _user_msg(image, _build_instruction(cot=False))]


# ----- P4: zero-shot, role, CoT -----
def p4_zero_role_cot(image):
    return [_system_msg(ROLE_SYSTEM), _user_msg(image, _build_instruction(cot=True))]


# ----- Few-shot variants P5-P8 (use later, not in pilot) -----
# These need 3 demo images. We'll wire those in once the zero-shot pilot looks good.

def _few_shot_preamble(demo_images, demo_labels):
    """
    Build the few-shot demonstration block. demo_images is a list of PIL Images,
    demo_labels is a list of class codes like ['mel','nv','bcc'].
    Returns a list of message dicts that go BEFORE the final query.
    """
    content = [{"type": "text",
                "text": "Here are 3 reference examples. Each is followed by its correct label."}]
    for img, lab in zip(demo_images, demo_labels):
        content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": f"Final answer: {lab}\n---"})
    content.append({"type": "text",
                    "text": "Now classify the next image using the same format."})
    return [{"role": "user", "content": content}]


def p5_few_norole_direct(image, demo_images, demo_labels):
    msgs = _few_shot_preamble(demo_images, demo_labels)
    msgs.append(_user_msg(image, _build_instruction(cot=False)))
    return msgs


def p6_few_norole_cot(image, demo_images, demo_labels):
    msgs = _few_shot_preamble(demo_images, demo_labels)
    msgs.append(_user_msg(image, _build_instruction(cot=True)))
    return msgs


def p7_few_role_direct(image, demo_images, demo_labels):
    msgs = [_system_msg(ROLE_SYSTEM)] + _few_shot_preamble(demo_images, demo_labels)
    msgs.append(_user_msg(image, _build_instruction(cot=False)))
    return msgs


def p8_few_role_cot(image, demo_images, demo_labels):
    msgs = [_system_msg(ROLE_SYSTEM)] + _few_shot_preamble(demo_images, demo_labels)
    msgs.append(_user_msg(image, _build_instruction(cot=True)))
    return msgs


# Convenience registry
ALL_PROMPTS = {
    "P1": p1_zero_norole_direct,
    "P2": p2_zero_norole_cot,
    "P3": p3_zero_role_direct,
    "P4": p4_zero_role_cot,
    "P5": p5_few_norole_direct,
    "P6": p6_few_norole_cot,
    "P7": p7_few_role_direct,
    "P8": p8_few_role_cot,
}

# For the pilot, we only use these two:
PILOT_PROMPTS = {
    "P1": p1_zero_norole_direct,
    "P4": p4_zero_role_cot,
}
