config = dict(
    epochs=100,
    batch_size=256,
    learning_rate_start=1e-3,
    epochs=100,
    betas=(1e-4, 0.02),
    num_timesteps=100,
    rul_max=125,
    dast_conf = {'input_size': 14, 'dec_seq_len': 4, 
                'dim_val_s': 64, 'dim_attn_s': 64, 'dim_val_t': 64, 'dim_attn_t': 64, 'dim_val': 64, 
                 'dim_attn': 64, 'n_decoder_layers': 1, 'n_encoder_layers': 2,
                 'n_heads': 4, 'dropout': 0.2}
)