model.py is the native LLaMA model file from https://github.com/Lightning-AI/lit-llama \\ 

WL.py, WL_S.py, WL_M.py are the Whispering LLaMA model artitectures built on top of model.py \\

adapter.py is the variant without the fusion adapters \\

big_adapter.py and ablation_mini_bad_init.py were used as part of the ablation experiments\\

WL.py, WL_S.py, WL_M.py have comments highlighting Key Modules (Adapters + Initilization Mechanism)