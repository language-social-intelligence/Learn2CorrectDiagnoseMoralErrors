"""Public chat contract and three-model-call alignment handoff."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from aiohttp import CookieJar
from aiohttp.test_utils import TestClient, TestServer
from assistant_pipeline import AuditedAssistant, LocalChatModel, base_dialogue_prompt, correction_prompt
from server import create_app
from test_assistant_pipeline import MainModel

CONFIG = json.loads((Path(__file__).parent/'assistant_config.json').read_text())


class Alignment:
    def diagnose(self, question, answer, mode):
        self.call = question, answer, mode
        return {'diagnosis':'An insult is present.','raw_output':'PRIVATE REVISION','active_adapter':mode}


class PipelineTests(unittest.TestCase):
    def test_light_is_default_and_only_diagnosis_reaches_final_correction(self):
        main = MainModel(['PRIVATE DRAFT','Final answer'])
        align = Alignment()
        stages = []
        result = AuditedAssistant(CONFIG, main=main, reviewers=align).generate({'question':'Question'},stages.append)
        self.assertEqual(align.call,('Question','PRIVATE DRAFT','light'))
        self.assertEqual(main.calls[1],(correction_prompt('Question','PRIVATE DRAFT','An insult is present.'),{'system':None}))
        self.assertNotIn('PRIVATE REVISION',main.calls[1][0])
        self.assertEqual(stages,['generating','diagnosing','paraphrasing','complete'])
        self.assertEqual(result['final_answer'],'Final answer')
        self.assertEqual(len(main.calls),2)

    def test_base_dialogue_uses_completion_without_instruct_tokens(self):
        prompt=base_dialogue_prompt('Explain gravity.')
        self.assertTrue(prompt.endswith('User: Explain gravity.\nAssistant: '))
        self.assertNotIn('[INST]',prompt)

    def test_final_rewrite_reaches_tokenizer_without_added_context(self):
        class StopAtTokenization(Exception):
            pass

        for question in ('Explain gravity.', '请解释重力。'):
            with self.subTest(question=question):
                model = LocalChatModel({**CONFIG['main_model'], 'prompt_mode': 'base_dialogue'})
                model.tokenizer = Mock(side_effect=StopAtTokenization)
                prompt = correction_prompt(question, 'Original reply.', 'No harmful content.')
                with self.assertRaises(StopAtTokenization):
                    model.chat(prompt, system=None)
                model.tokenizer.assert_called_once_with(prompt, return_tensors='pt')
                model.tokenizer.apply_chat_template.assert_not_called()

    def test_instruct_initial_chat_has_roles_and_one_bos(self):
        from transformers import AutoTokenizer
        model = LocalChatModel(CONFIG['main_model'])
        model.tokenizer = AutoTokenizer.from_pretrained(CONFIG['main_model']['checkpoint'], local_files_only=True)
        inputs = model.prepare_inputs('Explain gravity.')
        text = model.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        self.assertEqual(text.count('<|begin_of_text|>'), 1)
        self.assertIn('<|start_header_id|>user<|end_header_id|>\n\nExplain gravity.<|eot_id|>', text)
        self.assertTrue(text.endswith('<|start_header_id|>assistant<|end_header_id|>\n\n'))
        self.assertNotIn('What is photosynthesis?', text)

    def test_instruct_correction_prefill_is_in_assistant_turn(self):
        from transformers import AutoTokenizer
        model = LocalChatModel(CONFIG['main_model'])
        model.tokenizer = AutoTokenizer.from_pretrained(CONFIG['main_model']['checkpoint'], local_files_only=True)
        prompt = correction_prompt('Question', 'Draft', 'Diagnosis')
        inputs = model.prepare_inputs(prompt, system=None)
        text = model.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        expected = (
            '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'
            '<Prompt>: "Question"; <Reply>: "Draft". Here is a diagnosis.\n'
            '###Diagnosis: Diagnosis\n\n'
            'Based on the ###Diagnosis above, please rewrite the Reply.\n'
            'Only give the revised Reply without any explanation.<|eot_id|>\n'
            '<|start_header_id|>assistant<|end_header_id|>\n\nMy Revised Reply is: '
        )
        self.assertEqual(text, expected)


class Engine:
    def load(self): pass
    def generate(self, values, progress):
        self.last_values = dict(values)
        progress('paraphrasing')
        return {'final_answer':'Public answer','seconds':.1,'truncated':False,
                'diagnosis':'Private diagnosis','initial_answer':'Private draft','review_raw_output':'Private raw',
                'initial_input':'Private initial prompt','review_input':'Private review prompt',
                'correction_input':'Private correction prompt'}


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.config=copy.deepcopy(CONFIG)
        self.config['storage']['database']=str(Path(self.temp.name)/'requests.sqlite3')
        self.engine=Engine()
        self.client=TestClient(TestServer(create_app(self.config,self.engine)),cookie_jar=CookieJar(unsafe=True))
        await self.client.start_server()
        for _ in range(50):
            if (await (await self.client.get('/api/info')).json())['status']=='ready': break
            await asyncio.sleep(.01)

    async def asyncTearDown(self):
        await self.client.close()
        self.temp.cleanup()

    async def test_simple_request_only_exposes_final_answer(self):
        response=await self.client.post('/api/generate',json={'question':'Hello'})
        self.assertEqual(response.status,202)
        key=(await response.json())['id']
        for _ in range(50):
            data=await (await self.client.get('/api/results/'+key)).json()
            if data['status']=='complete':break
            await asyncio.sleep(.01)
        self.assertEqual(data['result'],{'final_answer':'Public answer','seconds':.1,'truncated':False})
        self.assertNotIn('Private',json.dumps(data))
        self.assertEqual(self.engine.last_values['review_mode'],'light')

    async def test_selected_mode_reaches_engine(self):
        info=await (await self.client.get('/api/info')).json()
        self.assertEqual([m['id'] for m in info['review_modes']],['light','fusion'])
        for mode in ('light','fusion'):
            response=await self.client.post('/api/generate',json={'question':'Hello','review_mode':mode})
            self.assertEqual(response.status,202)
            key=(await response.json())['id']
            for _ in range(50):
                data=await (await self.client.get('/api/results/'+key)).json()
                if data['status']=='complete':break
                await asyncio.sleep(.01)
            self.assertEqual(data['status'],'complete')
            self.assertEqual(self.engine.last_values['review_mode'],mode)

    async def test_two_chats_can_queue_without_removing_session_limit(self):
        import threading
        release=threading.Event()
        original=self.engine.generate
        def wait_then_generate(values,progress):
            release.wait(5)
            return original(values,progress)
        self.engine.generate=wait_then_generate
        try:
            for mode in ('fusion','light'):
                response=await self.client.post('/api/generate',json={'question':'Hello','review_mode':mode})
                self.assertEqual(response.status,202)
            response=await self.client.post('/api/generate',json={'question':'Third request','review_mode':'light'})
            self.assertEqual(response.status,429)
        finally:
            release.set()
        await asyncio.sleep(.05)

    async def test_named_route_and_no_arbitrary_file_access(self):
        for route in ('/','/human-scale-chatbot-education','/human-scale-chatbot-education/'):
            response=await self.client.get(route)
            self.assertEqual(response.status,200)
            self.assertIn('<title>Human-scale AI Chatbot</title>',await response.text())
        for route in ('/assistant_config.json','/legacy_assistant_config.json','/train.py','/data/second-look.sqlite3'):
            self.assertEqual((await self.client.get(route)).status,404)
        response=await self.client.post('/api/generate',json={'question':'Hi','review_mode':'unknown'})
        self.assertEqual(response.status,400)


if __name__=='__main__': unittest.main()
