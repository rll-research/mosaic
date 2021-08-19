# vary model sizes: attn{2x128, 2x256} x act{64-128-mix7, 64-128-mix14} x contras{128x256, 256x512}
python train.py use_all_tasks=True exp_name=7Task-TempContra set_same_n=1 bsize=61 vsize=61 \
    attn.n_attn_layers=2 attn.attn_ff=128 actions.n_layers=2 actions.out_dim=64 actions.hidden_dim=128 actions.n_mixtures=7 

# python train.py use_all_tasks=True exp_name=7Task-TempContra set_same_n=1 bsize=61 vsize=61 attn.n_attn_layers=2 attn.attn_ff=256 actions.n_layers=2 actions.out_dim=64 actions.hidden_dim=128 actions.n_mixtures=7; 
# python train.py use_all_tasks=True exp_name=7Task-TempContra set_same_n=1 bsize=61 vsize=61 attn.n_attn_layers=2 attn.attn_ff=256 actions.n_layers=2 actions.out_dim=64 actions.hidden_dim=128 actions.n_mixtures=14

# python train.py use_all_tasks=True exp_name=7Task-TempContra set_same_n=1 bsize=61 vsize=61 \
#     attn.n_attn_layers=2 attn.attn_ff=128 actions.n_layers=2 actions.out_dim=64 actions.hidden_dim=128 actions.n_mixtures=7 