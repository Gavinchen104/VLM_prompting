"""
prompts.py — All 8 prompt conditions for the HAM10000 pilot.

Each condition is a function that returns a `messages` list in the format
MedGemma expects. This format is the standard transformers chat-template
shape: a list of dicts with `role` and `content`, where `content` is itself
a list of typed parts (text/image).

For the pilot we only use P1 and P4 (most-different conditions).
The full 8 are here so you don't have to rewrite later.
"""

# Class vocabulary used in every prompt. Keep these consistent everywhere.
CLASSES_DESC = """- mel (melanoma)
- nv (melanocytic nevus)
- bcc (basal cell carcinoma)
- akiec (actinic keratosis / intraepithelial carcinoma)
- bkl (benign keratosis)
- df (dermatofibroma)
- vasc (vascular lesion)"""

CLASSES_SHORT = "mel, nv, bcc, akiec, bkl, df, vasc"


# ---------- Building blocks ----------

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
    instr = f"""Classify this dermoscopic image into exactly one of these classes:
{CLASSES_DESC}

Respond in exactly this format and nothing else:
Final answer: <class_code>"""
    return [_user_msg(image, instr)]


# ----- P2: zero-shot, no role, CoT -----
def p2_zero_norole_cot(image):
    instr = f"""Classify this dermoscopic image into exactly one of these classes:
{CLASSES_SHORT}.

Reason step by step before answering:
1. Describe what you see: asymmetry, border irregularity, color variation,
   diameter cues, and any dermoscopic structures (pigment network, globules,
   streaks, vessels).
2. List the top 2 candidate classes and why each fits or doesn't.
3. Commit to one class.

End your response with exactly this line:
Final answer: <class_code>"""
    return [_user_msg(image, instr)]


# ----- P3: zero-shot, role, direct -----
def p3_zero_role_direct(image):
    system = "You are an experienced dermatologist analyzing a dermoscopic image."
    instr = f"""Classify this image into exactly one of:
{CLASSES_SHORT}.

Respond in exactly this format and nothing else:
Final answer: <class_code>"""
    return [_system_msg(system), _user_msg(image, instr)]


# ----- P4: zero-shot, role, CoT -----
def p4_zero_role_cot(image):
    system = ("You are an experienced dermatologist analyzing a dermoscopic image. "
              "Use the ABCD rule and standard dermoscopic criteria.")
    instr = f"""Classify this image into exactly one of:
{CLASSES_SHORT}.

Reason step by step:
1. Asymmetry, Border, Color, Differential structures.
2. Top 2 candidate diagnoses with rationale.
3. Final commitment.

End your response with exactly this line:
Final answer: <class_code>"""
    return [_system_msg(system), _user_msg(image, instr)]


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
    msgs.append(_user_msg(image, f"""Classify this dermoscopic image into exactly one of:
{CLASSES_SHORT}.

Respond in exactly this format and nothing else:
Final answer: <class_code>"""))
    return msgs


def p6_few_norole_cot(image, demo_images, demo_labels):
    msgs = _few_shot_preamble(demo_images, demo_labels)
    msgs.append(_user_msg(image, f"""Classify this dermoscopic image into exactly one of:
{CLASSES_SHORT}.

Reason step by step (asymmetry/border/color/structures, top 2 candidates,
final commitment), then end with:
Final answer: <class_code>"""))
    return msgs


def p7_few_role_direct(image, demo_images, demo_labels):
    system = "You are an experienced dermatologist analyzing a dermoscopic image."
    msgs = [_system_msg(system)] + _few_shot_preamble(demo_images, demo_labels)
    msgs.append(_user_msg(image, f"""Classify this image into exactly one of:
{CLASSES_SHORT}.

Respond in exactly this format and nothing else:
Final answer: <class_code>"""))
    return msgs


def p8_few_role_cot(image, demo_images, demo_labels):
    system = ("You are an experienced dermatologist analyzing a dermoscopic image. "
              "Use the ABCD rule and standard dermoscopic criteria.")
    msgs = [_system_msg(system)] + _few_shot_preamble(demo_images, demo_labels)
    msgs.append(_user_msg(image, f"""Classify this image into exactly one of:
{CLASSES_SHORT}.

Reason step by step (ABCD rule, top 2 candidates, final commitment), then end with:
Final answer: <class_code>"""))
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
