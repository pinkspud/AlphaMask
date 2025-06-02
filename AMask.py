import torch
import folder_paths

class AddAlphaChannelFromMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_alpha"
    CATEGORY = "Custom/Image"

    def apply_alpha(self, images, mask, invert_mask):
        if isinstance(images, (tuple, list)):
            images = images[0]

        batch_size, h, w, c = images.shape

        if mask.shape != (h, w):
            raise ValueError(f"Mask size {mask.shape} doesn't match image size {(h, w)}")

        alpha = 1.0 - mask if invert_mask else mask

        alpha_stacked = alpha.unsqueeze(0).repeat(batch_size, 1, 1)
        result = torch.cat((images, alpha_stacked.unsqueeze(-1)), dim=-1)

        return (result,)

NODE_CLASS_MAPPINGS = {
    "AddAlphaChannelFromMask": AddAlphaChannelFromMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddAlphaChannelFromMask": "Add Alpha Channel From Mask"
}
