"""Check portable model validation and source packaging without loading weights."""
import json
from pathlib import Path
import tempfile
import unittest
from check_setup import ROOT, model_path, validate_model


class DeploymentTests(unittest.TestCase):
    def test_relative_paths_use_package_root(self):
        self.assertEqual(model_path('models/example'),ROOT/'models/example')

    def test_incomplete_sharded_copy_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'config.json').write_text('{"model_type":"muse_glimmer"}')
            (root/'tokenizer_config.json').write_text('{}')
            (root/'tokenizer.json').write_text('{}')
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'first':'model-1.safetensors','second':'model-2.safetensors'}}))
            (root/'model-1.safetensors').write_bytes(b'0'*1024)
            with self.assertRaisesRegex(ValueError,'model-2'):
                validate_model(directory,'muse_glimmer')
            (root/'model-2.safetensors').write_bytes(b'0'*1024)
            self.assertEqual(validate_model(directory,'muse_glimmer'),root)
            with self.assertRaisesRegex(ValueError,'expected model_type'):
                validate_model(directory,'mistral3')


if __name__ == '__main__':
    unittest.main()
