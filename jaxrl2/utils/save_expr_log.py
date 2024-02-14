def save_log(summary_writer, expr_log_dict, expr_step, expr_log_type, use_wandb=True):
    # use_wandb: if True, use wandb to log; if False, use tensorboard to log
    for k, v in expr_log_dict.items():
        if use_wandb:
            summary_writer.log({f"{expr_log_type}/{k}": v}, step=expr_step)
        else:
            if v.ndim == 0:
                summary_writer.add_scalar(f'{expr_log_type}/{k}', v, expr_step)
            else:
                summary_writer.add_histogram(f'{expr_log_type}/{k}', v, expr_step)
    if not use_wandb:
        summary_writer.flush()