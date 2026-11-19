import json
from pathlib import Path
import unittest

from assistant_pipeline import AuditedAssistant, correction_prompt, parse_light_diagnosis, parse_muse_answer

CONFIG = json.loads((Path(__file__).parent / "legacy_assistant_config.json").read_text())


class MainModel:
    def __init__(self, outputs, events=None):
        self.outputs = iter(outputs)
        self.calls = []
        self.events = events if events is not None else []

    def chat(self, question, **kwargs):
        self.calls.append((question, kwargs))
        self.events.append("main")
        return {"text": next(self.outputs), "tokens": 10, "truncated": False}


class LightModel:
    def __init__(self, diagnosis="The draft uses an insult.", revision="Replace the insult with a respectful disagreement.", events=None):
        self.diagnosis = diagnosis
        self.revision = revision
        self.calls = []
        self.events = events if events is not None else []

    def diagnose(self, question, answer):
        self.calls.append((question, answer))
        self.events.append("light")
        return {"diagnosis": self.diagnosis, "revision": self.revision, "raw_output": "light output"}


class PipelineTests(unittest.TestCase):
    def test_muse_reasoning_and_tool_channels_are_not_answers(self):
        raw = (' to=self<|message|>Private scratchpad.<|eom|>'
               '<|start|>assistant to=user<|message|>The answer.<|eot|>')
        self.assertEqual(parse_muse_answer(raw), 'The answer.')
        self.assertEqual(parse_muse_answer(' to=user<|message|>Direct answer.<|eot|>'), 'Direct answer.')
        self.assertEqual(parse_muse_answer('<|message|>Default recipient.<|eot|>'), 'Default recipient.')
        for unfinished in (' to=self<|message|>Still thinking', ' to=u',
                           ' to=search<|message|>Tool arguments<|eot|>'):
            with self.assertRaisesRegex(ValueError, 'did not reach a final answer'):
                parse_muse_answer(unfinished)

    def test_exact_three_stage_flow_discards_light_revision(self):
        events, stages = [], []
        main = MainModel(["Initial answer", "Final paraphrase"], events)
        light = LightModel(events=events)
        result = AuditedAssistant(CONFIG, main, light).generate({"question": "Original question"}, stages.append)
        self.assertEqual(events, ["main", "light", "main"])
        self.assertEqual(stages, ["generating", "diagnosing", "paraphrasing", "complete"])
        self.assertEqual(light.calls, [("Original question", "Initial answer")])
        self.assertEqual(main.calls[1], (
            '<Prompt>: "Original question"; <Reply>: "Initial answer". Here is a diagnosis.\n'
            '###Diagnosis: The draft uses an insult.\n\n'
            'Based on the ###Diagnosis above, please rewrite the Reply.\n'
            'Only give the revised Reply without any explanation.\n'
            'My Revised Reply is: ',
            {"system": None},
        ))
        self.assertNotIn(light.revision, main.calls[1][0])
        self.assertNotIn("light_revision", result)
        self.assertEqual(result["correction_method"], "diagnosis_only_single_pass")
        self.assertEqual(result["final_answer"], "Final paraphrase")
        self.assertNotIn("revision_audit", result)

    def test_no_change_guidance_still_goes_to_main_model(self):
        main = MainModel(["Useful original", "Useful final phrasing"])
        light = LightModel("No problematic cue identified.", "No substantive revision needed; preserve the answer.")
        result = AuditedAssistant(CONFIG, main, light).generate({"question": "Question"})
        self.assertEqual(len(main.calls), 2)
        self.assertEqual(len(light.calls), 1)
        self.assertEqual(result["final_answer"], "Useful final phrasing")
        self.assertNotIn(light.revision, main.calls[1][0])

    def test_diagnosis_without_revision_completes_one_rewrite(self):
        main = MainModel(["Initial answer", "Final answer"])
        result = AuditedAssistant(CONFIG, main, LightModel(revision="")).generate({"question": "Question"})
        self.assertEqual(len(main.calls), 2)
        self.assertEqual(result["final_answer"], "Final answer")

    def test_missing_diagnosis_prevents_rewrite(self):
        main = MainModel(["Initial answer"])
        with self.assertRaisesRegex(ValueError, "must provide a diagnosis"):
            AuditedAssistant(CONFIG, main, LightModel(diagnosis="")).generate({"question": "Question"})
        self.assertEqual(len(main.calls), 1)

    def test_numbered_diagnosis_discards_correction_step(self):
        raw = '(1) The draft insults the reader. (2) Remove the insult and address the question respectfully. The <Revised Reply> is: I disagree with this claim.'
        result = parse_light_diagnosis({"output": raw, "truncated": False})
        self.assertEqual(result["diagnosis"], "The draft insults the reader.")
        self.assertNotIn("revision", result)

    def test_explicit_revision_marker_is_supported(self):
        result = parse_light_diagnosis({"output": "###Diagnosis: An insult is present.\n###Revision: Replace it with a factual disagreement."})
        self.assertEqual(result["diagnosis"], "An insult is present.")
        self.assertNotIn("revision", result)

    def test_diagnosis_only_and_classification_explanations_are_supported(self):
        for raw in (
            '(1) An insult is present.',
            '(1) An insult is present. (2) The statement is toxic because of the insult. The utterance is toxic or benign? My answer is: toxic.',
        ):
            self.assertEqual(parse_light_diagnosis({"output": raw})["diagnosis"], "An insult is present.")

    def test_incomplete_or_classification_only_output_is_rejected(self):
        for raw in (
            {"output": ""},
            {"output": "(1) An insult. (2) Remove it.", "truncated": True},
            {"output": 'The utterance is toxic or benign? My answer is: toxic.'},
            {"output": 'toxic'},
        ):
            with self.assertRaises(ValueError):
                parse_light_diagnosis(raw)


if __name__ == "__main__":
    unittest.main()
