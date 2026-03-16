from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import torch
from torch import nn
from nnunetv2.architecetures.Swin_UMamba_2D import get_swin_umamba_2D_from_plans
from nnunetv2.architecetures.Swin_UMamba_3D import get_swin_umamba_3D_from_plans
# from typing import Tuple, Union, List
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from os.path import join


class nnUNetTrainerSwinUMamba(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4
        self.weight_decay = 5e-2
        self.enable_deep_supervision = True
        self.freeze_encoder_epochs = 10
        self.early_stop_epoch = 350

        # @staticmethod
        # def build_network_architecture(architecture_class_name: str,
        #                            arch_init_kwargs: dict,
        #                            arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        #                            num_input_channels: int,
        #                            num_output_channels: int,
        #                            enable_deep_supervision: bool = True) -> nn.Module:
            
        #     model = get_swin_umamba_2D_from_plans()
    def configure_optimizers(self):
        optimizer = AdamW(
            self.network.parameters(),
            lr=self.initial_lr, 
            weight_decay=self.weight_decay, 
            eps=1e-5,
            betas=(0.9, 0.999),
            )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-6)

        self.print_to_log_file(f"Using optimizer {optimizer}")
        self.print_to_log_file(f"Using scheduler {scheduler}")

        return optimizer, scheduler
    
    # def on_epoch_end(self):
    #     current_epoch = self.current_epoch
    #     if (current_epoch + 1) % self.save_every == 0:
    #         self.save_checkpoint(join(self.output_folder, f'checkpoint_{current_epoch}.pth'))
    #     super().on_epoch_end()

    def on_train_epoch_start(self):
        # freeze the encoder if the epoch is less than 10
        if self.current_epoch < self.freeze_encoder_epochs:
            self.print_to_log_file("Freezing the encoder")
            if self.is_ddp:
                self.network.module.freeze_encoder()
            else:
                self.network.freeze_encoder()
        else:
            self.print_to_log_file("Unfreezing the encoder")
            if self.is_ddp:
                self.network.module.unfreeze_encoder()
            else:
                self.network.unfreeze_encoder()
        super().on_train_epoch_start()

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            self.network.module.deep_supervision = enabled
        else:
            self.network.deep_supervision = enabled

    def _get_deep_supervision_scales(self):
        dims = len(self.configuration_manager.patch_size)
        if self.enable_deep_supervision:
            deep_supervision_scales = [[i]*dims for i in [1,0.5,0.25,0.125]]
        else:
            deep_supervision_scales = None  # for train and val_transforms
        return deep_supervision_scales