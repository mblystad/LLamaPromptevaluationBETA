"""
All prompt templates for different prompting strategies.
"""

from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# === TASK-ONLY PROMPT ===
stance_prompt = PromptTemplate(
    input_variables=["comment", "reply"],
    template="""Please classify the stance, or opinion, of the following reply to the comment.
Note that we want the stance of the reply to the comment, not to the topic of the comment.
Only give the stance as "agree", "disagree", or "neutral" and output no other words.

comment: {comment}
reply: {reply}
stance:"""
)

# === FEW-SHOT PROMPT ===
example_template = """comment: {comment}
reply: {reply}
stance: {stance}"""

example_prompt = PromptTemplate(
    input_variables=["comment", "reply", "stance"],
    template=example_template
)

few_shot_examples = [
    {'comment': "I think the new policy will help improve efficiency.",
     'reply': "I agree, it will make things more streamlined.",
     'stance': 'agree'},
    {'comment': "The new education reform seems promising.",
     'reply': "I disagree, it doesn't address the underlying issues.",
     'stance': 'disagree'},
    {'comment': "The park renovation project is a good idea.",
     'reply': "I’m not sure. It may be good, but the location is an issue.",
     'stance': 'neutral'},
    {'comment': "AI will revolutionize healthcare.",
     'reply': "I agree, it can save many lives.",
     'stance': 'agree'},
    {'comment': "The economy is recovering after the pandemic.",
     'reply': "I disagree, the recovery is too slow.",
     'stance': 'disagree'}
]

few_shot_prompt = FewShotPromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
    prefix="""Stance classification is the task of determining the expressed or implied opinion of a reply toward a comment.
The following replies express stances about comments.
Each reply must be classified as "agree", "disagree", or "neutral".""",
    suffix="""Analyze the following reply to the provided comment and determine its stance.
Return only a single word: "agree", "disagree", or "neutral".

comment: {comment}
reply: {reply}
stance:""",
    input_variables=["comment", "reply"],
    example_separator="\n"
)

# === CHAIN-OF-THOUGHT PROMPTS ===
cot_prompt_1 = PromptTemplate(
    input_variables=["comment", "reply"],
    template="""Stance classification is the task of determining the opinion of a reply toward a comment.
comment: {comment}
reply: {reply}
explanation:"""
)

cot_prompt_2 = PromptTemplate(
    input_variables=["comment", "reply", "stance_reason"],
    template="""Based on your explanation: "{stance_reason}", what is the final stance?
Only output one of the following: "agree", "disagree", or "neutral".

comment: {comment}
reply: {reply}
stance:"""
)

# === TREE-OF-THOUGHT PROMPTS ===
hypothesis_prompt = PromptTemplate(
    input_variables=["comment", "reply"],
    template="""Consider the following comment and reply:
comment: {comment}
reply: {reply}

Generate three different hypotheses for the stance of the reply toward the comment:
1. Agree
2. Disagree
3. Neutral

For each hypothesis, explain the reasoning behind that stance.
Label them clearly (e.g., "Hypothesis 1: ...")"""
)

evaluation_prompt = PromptTemplate(
    input_variables=["hypotheses", "comment", "reply"],
    template="""Evaluate the following hypotheses regarding the reply to the comment.

comment: {comment}
reply: {reply}

{hypotheses}

Score each hypothesis from 1 to 5 based on logic and relevance.
Format:
Hypothesis 1: 4, reason: ...
Hypothesis 2: 3, reason: ...
Only return scores and reasons."""
)

decision_prompt = PromptTemplate(
    input_variables=["hypotheses", "evaluations", "comment", "reply"],
    template="""Based on the hypotheses and evaluations below, choose the best stance.

comment: {comment}
reply: {reply}

{hypotheses}

{evaluations}

Return only one word: "agree", "disagree", or "neutral".
label:"""
)

# === SELF-CONSISTENCY PROMPT ===
consistency_prompt = PromptTemplate(
    input_variables=["comment", "reply", "tot_output", "cot_output", "few_shot_output", "task_output"],
    template="""Given the following comment and reply:

comment: {comment}
reply: {reply}

You have four stance outputs:
1. Tree-of-Thought: {tot_output}
2. Chain-of-Thought: {cot_output}
3. Few-Shot: {few_shot_output}
4. Task-Only: {task_output}

Which stance is most consistent across these techniques?
Return one word: "agree", "disagree", or "neutral" only.
"""
)
