# nmt_utils

Model and training code (plan sections 6–7), still to be written. `environment.py` already lives here, as in hung-asr-aws.

Planned: the composed model (frozen augmenting LLM → adapter prefix + gated cross-attention into the frozen anchor), the data collator (fixed prompt template, loss on target tokens only, target word dropout), and the three-stage trainer (bridge → LoRA SFT → CPO).
