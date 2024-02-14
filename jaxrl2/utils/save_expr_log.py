def save_log(summary_writer, expr_log_dict, expr_step, expr_log_type, use_wandb=True, decoder=None):
    # use_wandb: if True, use wandb to log; if False, use tensorboard to log
    for k, v in expr_log_dict.items():
        if decoder:
            k = decoder[k]
        if use_wandb:
            summary_writer.log({f"{expr_log_type}/{k}": v}, step=expr_step)
        else:
            # Assume v is an instance of a Python built-in type (int, float) or a numpy object (Array Scalar or ndarray)
            if (not hasattr(v, "ndim")) or (hasattr(v, "ndim") and (v.ndim == 0)):
                summary_writer.add_scalar(f'{expr_log_type}/{k}', v, expr_step)
            else:
                summary_writer.add_histogram(f'{expr_log_type}/{k}', v, expr_step)
    if not use_wandb:
        summary_writer.flush()