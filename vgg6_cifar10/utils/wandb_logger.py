
import wandb

def init_wandb(cfg):
    if not cfg.use_wandb:
        return None
    wandb.init(project=cfg.wandb_project,
               entity=cfg.wandb_entity,
               name=cfg.run_name,
               config=vars(cfg))
    return wandb

def finish_wandb(wandb_obj):
    if wandb_obj:
        wandb_obj.finish()
