# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A Ray logger will receive logging info from different processes.
"""
import json
import numbers
from pathlib import Path
from typing import Dict, Optional


def concat_dict_to_str(dict: Dict, step):
    output = [f'step:{step}']
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f'{k}:{v:.3f}')
    output_str = ' - '.join(output)
    return output_str


class LocalLogger:

    def __init__(self, remote_logger=None, enable_wandb=False, print_to_console=False, log_dir: Optional[str] = None):
        self.print_to_console = print_to_console
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_file = None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / 'training_metrics.jsonl'
            print(f'Saving training metrics to: {self.log_file}')
        
        if print_to_console:
            print('Using LocalLogger is deprecated. The constructor API will change ')

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)
        
        # Save to JSON Lines file if log_dir is specified
        if self.log_file:
            log_entry = {'step': step, **data}
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')