
'''
Copyright (c) 2021, Alibaba Cloud and its affiliates;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os

DEFAULT_WAVLM_MODEL = "patrickvonplaten/wavlm-libri-clean-100h-large"

MODEL_NAMES={
    "wavlm": os.environ.get("GLOBALDIFF_WAVLM_PATH", DEFAULT_WAVLM_MODEL),
}
